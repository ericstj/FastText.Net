# FastText.Net

A managed, **dependency-free** C# port of [Facebook fastText](https://github.com/facebookresearch/fastText)
**inference**, focused on text/language classification. Load a pretrained supervised
model (such as `lid.176.ftz` for language identification) and classify text without
shipping any native binaries.

> **Unofficial.** This is an independent, community port of the now-archived fastText
> project. It is **not affiliated with or endorsed by Meta / Facebook**. "fastText" is a
> trademark of its respective owner.

Only the prediction path is ported (model loading + `predict`); training is not included.

## Features

- Pure C# (`net10.0`), no native dependency, no P/Invoke.
- Reads the standard fastText binary model format, including **quantized** `.ftz`
  models (product quantization) and **hierarchical-softmax** / softmax / sigmoid losses.
- Byte-exact tokenization, FNV-1a hashing, and subword n-grams — results match the
  reference implementation (verified against the official Python `fasttext` package).
- SIMD-accelerated linear algebra via `System.Numerics.Tensors` (`TensorPrimitives`).
- Allocation-light prediction hot path (pooled buffers, thread-local state).

## Installation

```pwsh
dotnet add package FastText.Net
```

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

## Getting a model

Pretrained language-identification models are published by Facebook (CC BY-SA 3.0) and
are **not** bundled with this package:

- `lid.176.ftz` (~917 KB, quantized): https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz
- `lid.176.bin` (~126 MB, full): https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin

Any supervised fastText model in the standard binary format will load.

## License

MIT. Derived from fastText (MIT, © Facebook, Inc.). See the project repository for
full attribution and third-party notices: https://github.com/ericstj/FastText.Net
