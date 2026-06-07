# FastText.Net

A managed, **dependency-free** C# port of [Facebook fastText](https://github.com/facebookresearch/fastText)
**inference**, focused on text/language classification. Load a pretrained supervised
model (such as `lid.176.ftz` for language identification) and classify text without
shipping any native binaries.

This is a port of the prediction path only (model loading + `predict`); training is not
included. See [THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md) for attribution.

## Features

- Pure C# (`net10.0`), no native dependency, no P/Invoke.
- Reads the standard fastText binary model format, including **quantized** `.ftz`
  models (product quantization) and **hierarchical-softmax** / softmax / sigmoid losses.
- Byte-exact tokenization, FNV-1a hashing, and subword n-grams — results match the
  reference implementation (verified against the official Python `fasttext` package).
- SIMD-accelerated linear algebra via `System.Numerics.Tensors` (`TensorPrimitives`).
- Allocation-light prediction hot path (pooled buffers, thread-local state).

## Usage

```csharp
using FastTextNet;

var model = FastTextModel.Load("lid.176.ftz");

foreach (var p in model.Predict("Bonjour, comment allez-vous?", k: 3))
{
    Console.WriteLine($"{p.Label}: {p.Probability:P2}");
}
// __label__fr: 95.79%
// __label__en:  1.89%
// __label__nl:  0.26%
```

`FastTextModel` is immutable after loading and safe to share across threads.

### Getting a model

Language-identification models are published by Facebook (CC BY-SA 3.0) and are **not**
bundled here:

- `lid.176.ftz` (~917 KB, quantized): https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz
- `lid.176.bin` (~126 MB, full): https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin

## Layout

| Path | Description |
| --- | --- |
| `src/FastText.Net/` | The library. |
| `tests/FastText.Net.Tests/` | xUnit correctness tests vs. the reference `fasttext` outputs. |
| `bench/FastText.Net.Benchmarks/` | BenchmarkDotNet performance suite. |

## Building & testing

```pwsh
dotnet build -c Release
dotnet test  -c Release           # place lid.176.ftz under tests/.../models or set FASTTEXT_LID_MODEL
dotnet run   -c Release --project bench/FastText.Net.Benchmarks
```

## Benchmarks

Measured on the quantized `lid.176.ftz` model (16-dim embeddings, 176 labels) on a
single thread. See [bench/results.md](bench/results.md) for the full report and the
machine it was captured on.

## License

MIT — see [LICENSE](LICENSE). Derived from fastText (MIT, © Facebook, Inc.).
