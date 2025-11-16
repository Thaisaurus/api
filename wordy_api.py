import requests
import numpy as np
import os
from qdrant_client import QdrantClient, models
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from flask import Flask, request, jsonify
from flask_cors import CORS
from functools import lru_cache
from nltk.corpus import wordnet as wn

instruct_chat_endpoint = "http://127.0.0.1:8081/v1/chat/completions"
embedding_endpoint = "http://127.0.0.1:8080/embedding"
qdrant_host = "localhost"
qdrant_port = 6333
qdrant_collection = "thaisaurus-wordnet-qwen3embedding-nofallback-notags"#final-v1

prompt_modes = {
    "synonym": {
        "system": "You are an expert lexicographer. Your task is to write a single, concise sentence that describes the core meaning of a concept. Your response must be ONLY the single descriptive sentence, no markdown, no part of speech, no pronunciation, no examples, only the meaning.",
        "user": "Input: '{}'"
    },
    "antonym": {
        "system": "You are an expert lexicographer. Your task is to write a single, concise sentence that describes the core meaning of the OPPOSITE of a concept. Your response must be ONLY the single descriptive sentence, no markdown, no part of speech, no pronunciation, no examples, only the OPPOSITE meaning.",
        "user": "Describe the opposite concept of: '{}'"
    }
}

def normalize_l2(x):
    x = np.array(x)
    norm = np.linalg.norm(x)
    return x / norm if norm > 0 else x

@lru_cache(maxsize=2048)
def generate_query_definition(word: str, mode: str) -> str:
    prompt_template = prompt_modes.get(mode)
    if not prompt_template:
        raise ValueError(f"invalid mode: {mode}")

    payload = {
        "messages": [
            {"role": "system", "content": prompt_template["system"]},
            {"role": "user", "content": prompt_template["user"].format(word)}
        ],
        "stream": False, "temperature": 0.1, "n_predict": 128
    }
    try:
        response = requests.post(instruct_chat_endpoint, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()
        return data['choices'][0]['message']['content'].strip()
    except requests.exceptions.RequestException as e:
        print(f"error connecting to chat completion server: {e}")
        return None

@lru_cache(maxsize=2048)
def get_query_embedding(text_definition: str) -> list:
    payload = {"input": text_definition}
    try:
        response = requests.post(embedding_endpoint, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
        if isinstance(data, list) and data and isinstance(data[0], dict) and 'embedding' in data[0]:
            embedding = np.array(data[0]['embedding'][0], dtype=np.float32)
            return normalize_l2(embedding).tolist()
        else:
            print(f"error: unexpected response format: {str(data)[:200]}...")
            return None
    except requests.exceptions.RequestException as e:
        print(f"error connecting to the embedding server: {e}")
        return None

def get_vader_sentiment_label(word: str, sia: SentimentIntensityAnalyzer) -> str:
    score = sia.polarity_scores(word)['compound']
    if score >= 0.05: return "positive"
    elif score <= -0.05: return "negative"
    else: return "neutral"

def get_simple_pos(pos):
    if pos == 'n': return 'noun';
    if pos == 'v': return 'verb';
    if pos == 'a' or pos == 's': return 'adjective';
    if pos == 'r': return 'adverb';
    return 'other'

def perform_single_search(query_word: str, mode: str, pos_list: list, top_k: int, sia: SentimentIntensityAnalyzer) -> (list, str):
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
            query_definition = generate_query_definition(query_word, definition_mode)

        if not query_definition: return None, f"couldnt gen def from wordnet or LLM for: {query_word}."
        query_vector = get_query_embedding(query_definition)
        if not query_vector: return None, f"cailed gen embedding for def of: {query_word}."

        must_conditions = []
        query_word_sentiment = get_vader_sentiment_label(query_word, sia)

        if mode == "synonym":
            if query_word_sentiment == "positive": must_conditions.append(models.FieldCondition(key="sentiment_score", range=models.Range(gte=0.05)))
            elif query_word_sentiment == "negative": must_conditions.append(models.FieldCondition(key="sentiment_score", range=models.Range(lte=-0.05)))
        elif mode == "antonym":
            if query_word_sentiment == "positive": must_conditions.append(models.FieldCondition(key="sentiment_score", range=models.Range(lte=-0.05)))
            elif query_word_sentiment == "negative": must_conditions.append(models.FieldCondition(key="sentiment_score", range=models.Range(gte=0.05)))
        
        if pos_list:
            must_conditions.append(models.FieldCondition(key="pos", match=models.MatchAny(any=pos_list)))

        query_filter = models.Filter(must=must_conditions) if must_conditions else None
        search_results = qdrant_client.search(
            collection_name=qdrant_collection, query_vector=query_vector,
            query_filter=query_filter, limit=max(top_k*2,20)
        )

        distinct_results = filter_distinct_results(search_results, query_word)

        return distinct_results[:top_k], None
    except Exception as e:
        print(f"error during {mode} search for {query_word}: {e}")
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

def filter_distinct_results(raw_hits: list, query_phrase: str) -> list:
    best_hits = {}
    clean_query_phrase = query_phrase.strip().lower()
    for hit in raw_hits:
        word = hit.payload.get("word", "").strip().lower()
        if not word or word == clean_query_phrase:
            continue
        if word not in best_hits or hit.score > best_hits[word].score:
            best_hits[word] = hit
    return list(best_hits.values())

app = Flask(__name__)
CORS(app)

try:
    nltk.data.find('sentiment/vader_lexicon.zip')
    nltk.data.find('corpora/wordnet.zip')
    SIA = SentimentIntensityAnalyzer()
except LookupError:
    print("error: NLTK VADER lexicon not found")
    SIA = None

@app.route("/search", methods=["GET"])
def search():
    if not SIA: return jsonify({"error": "NLTK VADER lexicon not initialized on server."}), 500

    query_phrase = request.args.get("phrase", "angry")
    mode = request.args.get("mode", "search")
    top_k = request.args.get("n", default=10, type=int)
    pos_list = request.args.getlist("pos")
    width = request.args.get("width", type=int)
    height = request.args.get("height", type=int)

    if not width: return jsonify({"error": "i demand a width."}), 400
    if not height: return jsonify({"error": "i demand a height."}), 400

    raw_synonyms, raw_antonyms = [], []
    if mode == "synonym" or mode == "search":
        presults, error = perform_single_search(query_phrase, "synonym", pos_list, top_k, SIA)
        if error: return jsonify({"error": f"died on synonym search: {error}"}), 500
        if presults: raw_synonyms = presults
    if mode == "antonym" or mode == "search":
        presults, error = perform_single_search(query_phrase, "antonym", pos_list, top_k, SIA)
        if error: return jsonify({"error": f"died on antonym search: {error}"}), 500
        if presults: raw_antonyms = presults

    processed_synonyms = normalize_and_process_results(raw_synonyms, "synonym", width, height)
    processed_antonyms = normalize_and_process_results(raw_antonyms, "antonym", width, height)

    results = processed_synonyms + processed_antonyms

    return jsonify({"results": results})

if __name__ == "__main__":
    api_key = os.getenv("QDRANT_API_KEY")
    if not api_key:
        raise ValueError("QDRANT_API_KEY envvar not set")

    qdrant_host = os.getenv("QDRANT_HOST")
    if not qdrant_host:
        raise ValueError("QDRANT_HOST envvar not set")

    qdrant_client = QdrantClient(
        host=qdrant_host,
        port=qdrant_port,
        api_key=api_key,
        https=False
    )

    app.run(debug=True, port=5000)