# Third-Party Notices

This project, **fastText.Net**, is a managed C# port of the inference (model loading
and prediction) portion of **fastText**, an open-source library created by Facebook AI
Research (Meta).

## fastText

- Project: https://github.com/facebookresearch/fastText
- Copyright (c) 2016-present, Facebook, Inc.
- License: MIT

The C# source files in `src/FastText.Net/` are translated and adapted from the original
C++ sources (`args`, `dictionary`, `matrix`, `densematrix`, `quantmatrix`,
`productquantizer`, `model`, `loss`, and `fasttext`). The on-disk model binary format,
hashing (FNV-1a with signed-char semantics), subword tokenization, product-quantization
decoding, and hierarchical-softmax / softmax prediction algorithms are reproduced to
remain byte- and result-compatible with upstream fastText models (for example the
`lid.176` language-identification models).

### Original fastText MIT License

```
MIT License

Copyright (c) 2016-present, Facebook, Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Pretrained models

The `lid.176.bin` / `lid.176.ftz` language-identification models distributed by Facebook
are released under the
[Creative Commons Attribution-ShareAlike 3.0 (CC BY-SA 3.0)](https://creativecommons.org/licenses/by-sa/3.0/)
license and are **not** included in this repository. Download them from
https://fasttext.cc/docs/en/language-identification.html.
