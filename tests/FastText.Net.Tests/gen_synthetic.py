"""Trains tiny fastText supervised models in each loss mode and emits a matched
oracle, to validate FastText.Net's dense-matrix + softmax/sigmoid/hs prediction
paths (the ones lid.176.ftz does not exercise, since it is quantized + hs).

Models are intentionally tiny (dim=8, bucket=1000) so they can be committed as
regression fixtures. Each model file is paired with its own oracle predictions,
so cross-run training nondeterminism does not matter.
"""
import json
import os

import fasttext

fasttext.FastText.eprint = lambda *a, **k: None

HERE = os.path.dirname(__file__)
OUT_DIR = os.path.join(HERE, "synthetic")
os.makedirs(OUT_DIR, exist_ok=True)

# Small, clearly separable four-class corpus.
TRAIN = {
    "greeting": [
        "hello how are you today",
        "good morning everyone nice to meet you",
        "hi there how is it going",
        "hey welcome and good evening to you",
        "greetings friend how do you do",
        "good afternoon hope you are well",
    ],
    "weather": [
        "the rain and clouds bring a cold storm",
        "sunny skies with warm temperature today",
        "snow and ice cover the freezing mountains",
        "windy weather with thunder and lightning",
        "a humid forecast of fog and drizzle",
        "the dry heat wave continues this summer",
    ],
    "tech": [
        "the computer runs software and code fast",
        "machine learning models train on large data",
        "the network server processes many requests",
        "programming languages compile into binary",
        "the database stores and queries records",
        "cloud computing scales virtual machines",
    ],
    "food": [
        "the pizza and pasta taste delicious",
        "fresh bread butter and cheese for breakfast",
        "a sweet dessert of chocolate and cake",
        "grilled chicken with rice and vegetables",
        "the soup salad and sandwich for lunch",
        "spicy curry with noodles and tofu",
    ],
}

TEST_INPUTS = [
    "hello good morning how are you",
    "cold rain and snow with strong wind",
    "the software code runs on a fast computer",
    "delicious pizza pasta and sweet cake",
    "a server training machine learning models",
    "warm sunny weather today",
]

train_path = os.path.join(OUT_DIR, "train.txt")
with open(train_path, "w", encoding="utf-8") as f:
    for label, lines in TRAIN.items():
        for line in lines:
            f.write(f"__label__{label} {line}\n")

# (name, loss) combinations covering the untested prediction paths.
CONFIGS = [
    ("softmax", "softmax"),
    ("ns", "ns"),
    ("ova", "ova"),
    ("hs", "hs"),
]

manifest = []
for name, loss in CONFIGS:
    model = fasttext.train_supervised(
        input=train_path,
        loss=loss,
        dim=8,
        epoch=100,
        lr=0.5,
        wordNgrams=2,
        minn=2,
        maxn=4,
        bucket=1000,
        minCount=1,
        thread=1,
    )
    model_file = os.path.join(OUT_DIR, f"{name}.bin")
    model.save_model(model_file)

    nlabels = len(model.get_labels())
    cases = []
    for text in TEST_INPUTS:
        labels, probs = model.predict(text, k=nlabels)
        cases.append(
            {"text": text, "labels": list(labels), "probs": [float(p) for p in probs]}
        )
    manifest.append(
        {
            "model": f"{name}.bin",
            "loss": loss,
            "dim": model.get_dimension(),
            "labelCount": nlabels,
            "cases": cases,
        }
    )

os.remove(train_path)

with open(os.path.join(HERE, "synthetic_oracle.json"), "w", encoding="utf-8") as f:
    json.dump(manifest, f, ensure_ascii=False, indent=2)

sizes = {c["model"]: os.path.getsize(os.path.join(OUT_DIR, c["model"])) for c in manifest}
print("wrote synthetic_oracle.json and models:", sizes)
