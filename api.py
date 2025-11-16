from fastapi.responses import JSONResponse
import numpy as np
import os
from qdrant_client import AsyncQdrantClient, models
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import asyncio
import httpx
import traceback
from async_lru import alru_cache


# env, default or quit
def edq(key: str, default: str | None):
    v = os.getenv(key) or default
    if not v:
        print(f"missing env var {key}")
        exit(1)
    return v

instruct_chat_endpoint = edq("ENDPOINT_CHAT", "https://openrouter.ai/api/v1/chat/completions")
embedding_endpoint = edq("ENDPOINT_EMBEDDING", "https://openrouter.ai/api/v1/embeddings")
llm_model = edq("MODEL", "openai/gpt-oss-20b")
openrouter_api_key = edq("OPENROUTER_API_KEY", None)
qdrant_host = edq("QDRANT_HOST", "localhost")
qdrant_collection = edq("QDRANT_COLLECTION", "thaisaurus-wordnet-qwen3embedding-nofallback-notags") #final-v1
qdrant_api_key = edq("QDRANT_API_KEY", None)

http_client = httpx.AsyncClient(http2=True, headers={"Content-Type": "application/json", "Authorization": f"Bearer {openrouter_api_key}"})

qdrant_client = AsyncQdrantClient(
    host=qdrant_host,
    port=6333,
    api_key=qdrant_api_key,
    https=False,
)

prompt_modes = {
    "synonym": {
        "system": "You are an expert lexicographer. Your task is to write one or two concise sentences that describes the core meaning of a concept. Your response must be ONLY the descriptive sentence(s), no markdown, no part of speech, no pronunciation, no examples, only the meaning.",
        "user": "Input: '{}'"
    },
    "antonym": {
        "system": "You are an expert lexicographer. Your task is to write one or two concise sentences that describes the core meaning of the OPPOSITE of a concept. Your response must be ONLY the descriptive sentence(s), no markdown, no part of speech, no pronunciation, no examples, only the OPPOSITE meaning.",
        "user": "Describe the opposite concept of: '{}'"
    }
}

def normalize_l2(x):
    x = np.array(x)
    norm = np.linalg.norm(x)
    return x / norm if norm > 0 else x

@alru_cache(maxsize=2048)
async def generate_query_definition(word: str, mode: str) -> str:
    prompt_template = prompt_modes.get(mode)
    if not prompt_template:
        raise ValueError(f"invalid mode: {mode}")

    payload = {
        "model": llm_model,
        "messages": [
            {"role": "system", "content": prompt_template["system"]},
            {"role": "user", "content": prompt_template["user"].format(word)}
        ],
        "extra_body": {"reasoning": {"enabled": False}},
        "stream": False, "temperature": 0.1, "n_predict": 128
    }
    try:
        response = await http_client.post(instruct_chat_endpoint, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()
        return data['choices'][0]['message']['content'].strip()
    except httpx.RequestError as e:
        print(f"error connecting to chat completion server: {e}")
        return None

@alru_cache(maxsize=2048)
async def get_query_embedding(text_definition: str) -> list:
    payload = {"input": text_definition, "model": "qwen/qwen3-embedding-4b", "encoding_format": "float"}
    try:
        response = await http_client.post(embedding_endpoint, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()['data']
        if isinstance(data, list) and data and isinstance(data[0], dict) and 'embedding' in data[0]:
            embedding = np.array(data[0]['embedding'], dtype=np.float32)
            return normalize_l2(embedding).tolist()
        else:
            print(f"error: unexpected response format: {str(data)[:200]}...")
            return None
    except httpx.RequestError as e:
        print(f"error connecting to the embedding server: {e}")
        return None

async def get_vader_sentiment_label(word: str, sia: SentimentIntensityAnalyzer) -> str:
    scores = await asyncio.to_thread(sia.polarity_scores, word)
    score = scores['compound']
    if score >= 0.05: return "positive"
    elif score <= -0.05: return "negative"
    else: return "neutral"

def get_simple_pos(pos):
    if pos == 'n': return 'noun';
    if pos == 'v': return 'verb';
    if pos == 'a' or pos == 's': return 'adjective';
    if pos == 'r': return 'adverb';
    return 'other'

async def perform_single_search(query_word: str, mode: str, pos_list: list, top_k: int, sia: SentimentIntensityAnalyzer) -> (list, str):
    try:
        query_definition = None

        if mode == "synonym" and (" " not in query_word.strip()):
            synsets = wn.synsets(query_word)
            if synsets:
                if pos_list:
                    for synset in synsets:
                        synset_pos = get_simple_pos(synset.pos())
                        if synset_pos in pos_list:
                            query_definition = synset.definition()
                            break

                if not query_definition:
                    query_definition = synsets[0].definition()

        if not query_definition:
            definition_mode = "antonym" if mode == "antonym" else "synonym"
            query_definition = await generate_query_definition(query_word, definition_mode)

        if not query_definition: return None, f"couldnt gen def from wordnet or LLM for: {query_word}."
        query_vector = await get_query_embedding(query_definition)
        if not query_vector: return None, f"failed gen embedding for def of: {query_word}."

        must_conditions = []
        query_word_sentiment = await get_vader_sentiment_label(query_word, sia)

        if mode == "synonym":
            if query_word_sentiment == "positive": must_conditions.append(models.FieldCondition(key="sentiment_score", range=models.Range(gte=0.05)))
            elif query_word_sentiment == "negative": must_conditions.append(models.FieldCondition(key="sentiment_score", range=models.Range(lte=-0.05)))
        elif mode == "antonym":
            if query_word_sentiment == "positive": must_conditions.append(models.FieldCondition(key="sentiment_score", range=models.Range(lte=-0.05)))
            elif query_word_sentiment == "negative": must_conditions.append(models.FieldCondition(key="sentiment_score", range=models.Range(gte=0.05)))
        
        if pos_list:
            must_conditions.append(models.FieldCondition(key="pos", match=models.MatchAny(any=pos_list)))

        query_filter = models.Filter(must=must_conditions) if must_conditions else None
        search_results = await qdrant_client.search(
            collection_name=qdrant_collection, query_vector=query_vector,
            query_filter=query_filter, limit=max(top_k*3,20)
        )

        return search_results, None
    except Exception as e:
        print(f"error during {mode} search for {query_word}: {e}")
        print(traceback.format_exc())
        return None, "internal error of some kind"

def lerp(start, end, t):
    return start + t * (end - start)

def lerp_color_oklch(color_start, color_end, factor):
    t = max(0.0, min(1.0, factor))
    l = lerp(color_start[0], color_end[0], t)
    c = lerp(color_start[1], color_end[1], t)
    h = lerp(color_start[2], color_end[2], t)
    return {"l": l, "c": c, "h": h}

def normalize_and_process_results(raw_hits: list, mode: str, width: int, height: int) -> list:
    if not raw_hits:
        return []

    scores = [hit.score for hit in raw_hits]
    min_score, max_score = min(scores), max(scores)
    score_range = max_score - min_score

    color_white = (1.0, 0.0, 0.0)
    if mode == "synonym":
        end_color_display = (0.8559, 0.233, 130.85)
    elif mode == "antonym":
        end_color_display = (0.592, 0.2026, 22.97)
    else:
        end_color_display = (0.828, 0.189, 84.429)

    processed_results = []
    normalized_scores = []

    for hit in raw_hits:
        if score_range == 0:
            normalized_similarity = 1.0
        else:
            normalized_similarity = (hit.score - min_score) / score_range
        normalized_scores.append(normalized_similarity)
        
        processed_results.append({
            "id": hit.id,
            "phrase": {"content": hit.payload.get("word", ""), "definition": hit.payload.get("definition", "")},
            "similarity": hit.score,
            "sentiment_score": hit.payload.get("sentiment_score", 0.0),
            "pos": hit.payload.get("pos", "other"),
            "tags": [],
            "wordClass": mode,
            "color": lerp_color_oklch(color_white, end_color_display, normalized_similarity),
            "position": {"x": 0, "y": 0}
        })

    center_point = np.array([50.0, 50.0])
    normalized_radius = 50.0
    
    if width >= height and height > 0:
        scale_x = 1.0
        scale_y = height / width
    elif height > width and width > 0:
        scale_x = width / height
        scale_y = 1.0
    else:
        scale_x, scale_y = 1.0, 1.0

    effective_radius_x = normalized_radius * scale_x
    effective_radius_y = normalized_radius * scale_y

    n = len(raw_hits)
    rotation_offset = -1.5 * np.pi / 6
    theta_base = np.linspace(np.pi / 6, 5 * np.pi / 6, n) if n > 1 else np.array([np.pi / 2])
    
    s_normalized = np.array(normalized_scores)

    if mode == "synonym" or mode == "search":
        abstract_radius_scale = (13 * ((1 - s_normalized) ** 0.5) + 10) / 50.0
        thetas = theta_base + rotation_offset
        offsets_x = effective_radius_x * abstract_radius_scale * np.cos(thetas)
        offsets_y = effective_radius_y * abstract_radius_scale * -np.sin(thetas)
    elif mode == "antonym":
        abstract_radius_scale = (13 * ((1 - s_normalized) ** 0.5) + 10) / 50.0
        thetas = -theta_base + rotation_offset
        offsets_x = effective_radius_x * abstract_radius_scale * np.sin(thetas)
        offsets_y = effective_radius_y * abstract_radius_scale * -np.cos(thetas)

    for i, res in enumerate(processed_results):
        res['position']['x'] = center_point[0] + offsets_x[i]
        res['position']['y'] = center_point[1] + offsets_y[i]

    return processed_results

def get_wordnet_pos(pos: str):
    if pos == 'noun': return wn.NOUN
    if pos == 'verb': return wn.VERB
    if pos == 'adjective': return wn.ADJ
    if pos == 'adverb': return wn.ADV
    return wn.NOUN

def filter_distinct_results(raw_hits: list, query_phrase: str, lemmatizer: WordNetLemmatizer) -> list:
    if not raw_hits:
        return []

    exclusion_set = set()
    clean_query_phrase = query_phrase.strip().lower()

    exclusion_set.add(clean_query_phrase)

    for synset in wn.synsets(clean_query_phrase):
        for lemma in synset.lemmas():
            exclusion_set.add(lemma.name().lower())
            for related_form in lemma.derivationally_related_forms():
                exclusion_set.add(related_form.name().lower())

    for word_root in list(exclusion_set):
        exclusion_set.add(word_root + "s")
        exclusion_set.add(word_root + "es")
        if word_root.endswith('y'):
             exclusion_set.add(word_root[:-1] + "ily")
        else:
             exclusion_set.add(word_root + "ly")
        exclusion_set.add(word_root + "ness")
        exclusion_set.add(word_root + "er")
        exclusion_set.add(word_root + "est")

    best_hits_by_lemma = {}

    for hit in raw_hits:
        word = hit.payload.get("word", "").strip().lower()

        if not word or word in exclusion_set:
            continue

        pos_tag = get_wordnet_pos(hit.payload.get("pos", "noun"))
        lemma = lemmatizer.lemmatize(word, pos=pos_tag)

        if lemma not in best_hits_by_lemma:
            best_hits_by_lemma[lemma] = hit

    return list(best_hits_by_lemma.values())

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

try:
    nltk.data.find('sentiment/vader_lexicon.zip')
    nltk.data.find('corpora/wordnet.zip')
    SIA = SentimentIntensityAnalyzer()
    LEMMATIZER = WordNetLemmatizer()
except LookupError:
    print("error: NLTK VADER lexicon not found")
    SIA = None

@app.get("/search")
async def search(width: int, height: int, phrase: str = "angry", mode: str = "search", n: int = 10, pos: list[str] = Query(None)):
    if not SIA: return JSONResponse({"error": "NLTK VADER lexicon not initialized on server."}, 500)

    query_phrase = phrase
    top_k = n
    pos_list = pos

    if not width: return JSONResponse({"error": "i demand a width."}, 400)
    if not height: return JSONResponse({"error": "i demand a height."}, 400)

    raw_synonyms, raw_antonyms = [], []
    if mode == "synonym" or mode == "search":
        presults, error = await perform_single_search(query_phrase, "synonym", pos_list, top_k, SIA)
        if error: return JSONResponse({"error": f"died on synonym search: {error}"}, 500)
        if presults: raw_synonyms = presults
    if mode == "antonym" or mode == "search":
        presults, error = await perform_single_search(query_phrase, "antonym", pos_list, top_k, SIA)
        if error: return JSONResponse({"error": f"died on antonym search: {error}"}, 500)
        if presults: raw_antonyms = presults

    raw_hits = raw_synonyms + raw_antonyms
    distinct_hits = filter_distinct_results(raw_hits, query_phrase, LEMMATIZER)
    final_hits = distinct_hits[:top_k]

    final_synonyms = [h for h in final_hits if h.id in {hit.id for hit in raw_synonyms}]
    final_antonyms = [h for h in final_hits if h.id in {hit.id for hit in raw_antonyms}]

    processed_synonyms = normalize_and_process_results(final_synonyms, "synonym", width, height)
    processed_antonyms = normalize_and_process_results(final_antonyms, "antonym", width, height)

    results = processed_synonyms + processed_antonyms

    return JSONResponse({"results": results})

if __name__ == "__main__":
    uvicorn.run("api:app", port=5000, host="0.0.0.0")
