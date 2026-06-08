"""Emits a Phase 1 inference oracle (word/subword/sentence vectors, ids, and
nearest neighbours) from the EXISTING committed synthetic models. Does not retrain,
so the .bin fixtures and their prediction oracles stay byte-identical.

Validates FastText.Net's new full-surface FastText facade against reference fastText.
"""
import json
import os

import fasttext

fasttext.FastText.eprint = lambda *a, **k: None

HERE = os.path.dirname(__file__)
SYNTH_DIR = os.path.join(HERE, "synthetic")

MODELS = ["softmax.bin", "ns.bin", "ova.bin", "hs.bin"]

# Probe words chosen from the synthetic corpus vocabulary.
PROBE_WORDS = ["hello", "weather", "computer", "pizza", "the"]
PROBE_SUBWORDS = ["hel", "comp", "izza"]
SENTENCES = [
    "hello good morning how are you",
    "the software code runs on a fast computer",
    "delicious pizza pasta and sweet cake",
]


def vec(a):
    return [float(x) for x in a]


manifest = []
for model_file in MODELS:
    m = fasttext.load_model(os.path.join(SYNTH_DIR, model_file))
    vocab = set(m.get_words())

    probes = [w for w in PROBE_WORDS if w in vocab]

    word_vectors = {w: vec(m.get_word_vector(w)) for w in probes}
    word_ids = {w: int(m.get_word_id(w)) for w in probes}
    subword_ids = {s: int(m.get_subword_id(s)) for s in PROBE_SUBWORDS}
    subword_vectors = {
        s: vec(m.get_input_vector(m.get_subword_id(s))) for s in PROBE_SUBWORDS
    }
    sentence_vectors = {s: vec(m.get_sentence_vector(s)) for s in SENTENCES}

    nn = {}
    for w in probes:
        neighbours = m.get_nearest_neighbors(w, k=5)
        nn[w] = [{"word": word, "sim": float(sim)} for sim, word in neighbours]

    manifest.append(
        {
            "model": model_file,
            "dim": m.get_dimension(),
            "wordVectors": word_vectors,
            "wordIds": word_ids,
            "subwordIds": subword_ids,
            "subwordVectors": subword_vectors,
            "sentenceVectors": sentence_vectors,
            "nearestNeighbors": nn,
        }
    )

with open(os.path.join(HERE, "vectors_oracle.json"), "w", encoding="utf-8") as f:
    json.dump(manifest, f, ensure_ascii=False, indent=2)

print("wrote vectors_oracle.json for", [c["model"] for c in manifest])
