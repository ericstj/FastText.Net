# fasttext (FastText.Net.Cli)

A managed command-line port of the
[Facebook fastText](https://github.com/facebookresearch/fastText) tool. Train, evaluate,
quantize, and query text-classification and word-embedding models from the terminal — with
no native binaries and no P/Invoke.

> **Unofficial.** This is an independent, community port of the now-archived fastText
> project. It is **not affiliated with or endorsed by Meta / Facebook**. "fastText" is a
> trademark of its respective owner.

## Installation

This is a framework-dependent [.NET tool](https://learn.microsoft.com/dotnet/core/tools/global-tools),
so it needs the [.NET runtime](https://dotnet.microsoft.com/download) (net10.0 or later).

Install it globally:

```pwsh
dotnet tool install -g FastText.Net.Cli
```

The command is `fasttext`:

```pwsh
fasttext supervised -input train.txt -output model
```

## Commands

The command surface mirrors the original `fasttext` tool:

| Command | Description |
| --- | --- |
| `supervised` | Train a supervised classifier |
| `skipgram` / `cbow` | Train word embeddings |
| `test` / `test-label` | Evaluate a classifier (overall / per-label) |
| `quantize` | Quantize a model to reduce its size |
| `predict` / `predict-prob` | Predict labels (optionally with probabilities) |
| `print-word-vectors` | Print word vectors read from stdin |
| `print-sentence-vectors` | Print sentence vectors read from stdin |
| `print-ngrams` | Print subword n-gram vectors for a word |
| `nn` | Query nearest neighbors |
| `analogies` | Query word analogies |
| `dump` | Dump `args`, `dict`, `input`, or `output` |

Run `fasttext` with no arguments for the full usage list.

## Examples

Train and evaluate a classifier:

```pwsh
fasttext supervised -input train.txt -output model -epoch 25 -lr 1.0
fasttext test model.bin test.txt
echo "which language is this" | fasttext predict-prob model.bin - 3
```

Train and explore word embeddings:

```pwsh
fasttext skipgram -input corpus.txt -output vectors
echo "king" | fasttext print-word-vectors vectors.bin
```

Input files use the fastText format: one example per line, with labels prefixed by
`__label__` (the prefix is configurable via `-label`).

## License

MIT. Derived from fastText (MIT, © Facebook, Inc.). See the project repository for full
attribution and third-party notices: https://github.com/ericstj/FastText.Net
