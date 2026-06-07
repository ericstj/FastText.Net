import json, os, sys
import fasttext
fasttext.FastText.eprint = lambda *a, **k: None

model_path = sys.argv[1]
m = fasttext.load_model(model_path)

sentences = [
    "Hello, how are you doing today?",
    "Bonjour, comment allez-vous aujourd'hui?",
    "Hola, ¿cómo estás hoy?",
    "Guten Tag, wie geht es Ihnen heute?",
    "Ciao, come stai oggi?",
    "Привет, как у тебя дела сегодня?",
    "こんにちは、お元気ですか",
    "你好，今天过得怎么样",
    "안녕하세요 오늘 기분이 어떠세요",
    "مرحبا كيف حالك اليوم",
    "Olá, como você está hoje?",
    "Hej, hur mår du idag?",
    "The quick brown fox jumps over the lazy dog",
    "Performance benchmarking of machine learning inference",
    "fastText is a library for efficient text classification",
    "a",
    "123 456 789",
]

out = []
for s in sentences:
    labels, probs = m.predict(s, k=5)
    out.append({"text": s, "labels": list(labels), "probs": [float(p) for p in probs]})

with open(os.path.join(os.path.dirname(__file__), "oracle_lid.json"), "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
print("wrote oracle_lid.json with", len(out), "entries")
