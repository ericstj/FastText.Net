# Benchmark results

Captured with [BenchmarkDotNet](https://benchmarkdotnet.org/) over the quantized
`lid.176.ftz` language-identification model (16-dim embeddings, 176 labels,
hierarchical softmax, product-quantized input). Single-threaded.

To reproduce:

```pwsh
dotnet run -c Release --project bench/FastText.Net.Benchmarks -- --filter *
```

## Environment

```
BenchmarkDotNet v0.14.0, Windows 11 (10.0.26200.8390)
13th Gen Intel Core i7-13800H, 1 CPU, 20 logical and 14 physical cores
.NET SDK 11.0.100-preview.4.26208.110
  [Host]     : .NET 10.0.8 (10.0.826.23019), X64 RyuJIT AVX2
  DefaultJob : .NET 10.0.8 (10.0.826.23019), X64 RyuJIT AVX2
```

## Prediction

| Method           | Mean        | Error     | StdDev    | Ratio | Gen0   | Allocated |
|----------------- |------------:|----------:|----------:|------:|-------:|----------:|
| PredictShort_K1  |    846.9 ns |  31.46 ns |  92.76 ns |  1.00 | 0.0172 |     216 B |
| PredictShort_K5  |  3,272.1 ns |  65.25 ns |  67.01 ns |  3.91 | 0.0343 |     472 B |
| PredictLong_K1   |  4,018.1 ns |  47.24 ns |  41.88 ns |  4.80 | 0.0343 |     456 B |
| PredictCorpus_K1 | 15,387.7 ns | 295.30 ns | 432.84 ns | 18.38 | 0.1526 |    2048 B |

- **PredictShort_K1** — top-1 language for a short sentence: **~847 ns (~1.18M predictions/sec)**, 216 B allocated.
- **PredictShort_K5** — top-5 for the same input.
- **PredictLong_K1** — top-1 for a longer paragraph.
- **PredictCorpus_K1** — top-1 over a multi-sentence corpus string.

## Model load

| Method    | Mean     | Error     | StdDev    |
|---------- |---------:|----------:|----------:|
| LoadModel | 4.324 ms | 0.0731 ms | 0.0750 ms |

Loading and fully decoding `lid.176.ftz` from disk takes **~4.3 ms**. Loading is a
one-time cost; the resulting `FastTextModel` is immutable and reused across predictions.
