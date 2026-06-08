// Ported from Facebook fastText (src/fasttext.h, src/fasttext.cc).
// See THIRD-PARTY-NOTICES.md. Original: Copyright (c) Facebook, Inc. (MIT License).

using System.Text;

namespace FastTextNet;

/// <summary>A word returned by a nearest-neighbor or analogy query, with its cosine similarity.</summary>
public readonly record struct FastTextNeighbor(string Word, float Similarity);

/// <summary>A subword n-gram and its input vector, as returned by <see cref="FastText.GetNgramVectors"/>.</summary>
public readonly record struct FastTextNgramVector(string Ngram, float[] Vector);

/// <summary>
/// Managed equivalent of the native fastText object, exposing the full inference surface
/// (word/subword/sentence vectors, nearest neighbors, analogies, and prediction).
/// </summary>
public sealed partial class FastText
{
    private const float SimilarityEpsilon = 1e-8f;

    private readonly Args _args;
    private readonly Dictionary _dict;
    private Matrix _input;
    private Matrix _output;
    private Model _model;
    private bool _quantized;

    private DenseMatrix? _precomputedWordVectors;

    private FastText(LoadedModel m)
    {
        _args = m.Args;
        _dict = m.Dict;
        _input = m.Input;
        _output = m.Output;
        _model = m.Model;
        _quantized = m.Quantized;
    }

    private FastText(Args args, Dictionary dict, Matrix input, Matrix output, Model model)
    {
        _args = args;
        _dict = dict;
        _input = input;
        _output = output;
        _model = model;
        _quantized = false;
    }

    /// <summary>Loads a model from a file path.</summary>
    public static FastText LoadModel(string path)
    {
        using var stream = new FileStream(
            path, FileMode.Open, FileAccess.Read, FileShare.Read, 1 << 20, FileOptions.SequentialScan);
        return LoadModel(stream);
    }

    /// <summary>Loads a model from a stream.</summary>
    public static FastText LoadModel(Stream stream)
    {
        ArgumentNullException.ThrowIfNull(stream);
        return new FastText(ModelLoader.Load(stream));
    }

    /// <summary>Saves the model to a file path in fastText binary format.</summary>
    public void SaveModel(string path)
    {
        ArgumentNullException.ThrowIfNull(path);
        using var stream = new FileStream(
            path, FileMode.Create, FileAccess.Write, FileShare.None, 1 << 20, FileOptions.SequentialScan);
        SaveModel(stream);
    }

    /// <summary>Saves the model to a stream in fastText binary format.</summary>
    public void SaveModel(Stream stream)
    {
        ArgumentNullException.ThrowIfNull(stream);
        ModelLoader.Save(stream, _args, _dict, _input, _output, _quantized);
    }

    /// <summary>Writes word vectors in the text <c>.vec</c> format (header line then one word per line).</summary>
    public void SaveVectors(string path)
    {
        ArgumentNullException.ThrowIfNull(path);
        using var writer = new StreamWriter(path, append: false, Encoding.UTF8);
        SaveVectors(writer);
    }

    /// <summary>Writes word vectors in the text <c>.vec</c> format to a writer.</summary>
    public void SaveVectors(TextWriter writer)
    {
        ArgumentNullException.ThrowIfNull(writer);
        int nwords = _dict.NWords;
        writer.Write(nwords);
        writer.Write(' ');
        writer.Write(_args.Dim);
        writer.Write('\n');

        var vec = new Vector(_args.Dim);
        for (int i = 0; i < nwords; i++)
        {
            AddWordVector(vec, _dict.GetWordBytes(i));
            writer.Write(_dict.GetWord(i));
            WriteVector(writer, vec.Data);
        }
    }

    /// <summary>Writes the output (label/word) vectors in text format. Not supported for quantized models.</summary>
    public void SaveOutput(string path)
    {
        ArgumentNullException.ThrowIfNull(path);
        using var writer = new StreamWriter(path, append: false, Encoding.UTF8);
        SaveOutput(writer);
    }

    /// <summary>Writes the output (label/word) vectors in text format to a writer.</summary>
    public void SaveOutput(TextWriter writer)
    {
        ArgumentNullException.ThrowIfNull(writer);
        if (_quantized)
        {
            throw new InvalidOperationException("SaveOutput is not supported for quantized models.");
        }
        int n = _args.Model == ModelName.Sup ? _dict.NLabels : _dict.NWords;
        writer.Write(n);
        writer.Write(' ');
        writer.Write(_args.Dim);
        writer.Write('\n');

        var vec = new Vector(_args.Dim);
        for (int i = 0; i < n; i++)
        {
            vec.Zero();
            vec.AddRow(_output, i);
            writer.Write(_args.Model == ModelName.Sup ? _dict.GetLabel(i) : _dict.GetWord(i));
            WriteVector(writer, vec.Data);
        }
    }

    private static void WriteVector(TextWriter writer, ReadOnlySpan<float> data)
    {
        foreach (float value in data)
        {
            writer.Write(' ');
            writer.Write(value.ToString("g6", System.Globalization.CultureInfo.InvariantCulture));
        }
        writer.Write('\n');
    }

    /// <summary>Embedding dimension of the model.</summary>
    public int Dimension => _args.Dim;

    /// <summary>True if the model uses product quantization (a <c>.ftz</c> model).</summary>
    public bool IsQuantized => _quantized;

    /// <summary>True if the model is supervised (classification), false for cbow/skipgram.</summary>
    public bool IsSupervised => _args.Model == ModelName.Sup;

    /// <summary>
    /// Returns the output-matrix index for a label string, or -1 if the label is unknown.
    /// Mirrors fastText's <c>getLabelId</c>.
    /// </summary>
    public int GetLabelId(string label)
    {
        ArgumentNullException.ThrowIfNull(label);
        int id = _dict.GetId(System.Text.Encoding.UTF8.GetBytes(label));
        return id == -1 ? -1 : id - _dict.NWords;
    }

    // Quantizes in place to satisfy an autotune model-size budget, mirroring Autotune::quantize.
    // Returns false when the size constraint provably cannot be met (cutoff hits the floor).
    internal bool TryQuantizeForSize(int dsub, long targetFileSize)
    {
        const int cutoffLimit = 256;
        long outM = _output.Rows;
        long outN = _output.Cols;
        long dim = _input.Cols;

        bool qnorm = true;
        bool qout = outM >= cutoffLimit;

        long outModelSize = qout
            ? 21 + outM * ((outN + 2 - 1) / 2) + (16 + 4 * (outN * (1 << 8))) + outM
            : 16 + 4 * (outM * outN);

        long target = targetFileSize - 107 - 4 * (1 << 8) * dim - outModelSize;
        long denom = (dim + dsub - 1) / dsub + 1 + 10;
        int cutoff = (int)Math.Max(target / denom, cutoffLimit);

        if (cutoff == cutoffLimit)
        {
            return false;
        }

        Quantize(dsub, qnorm, cutoff, qout);
        return true;
    }

    /// <summary>
    /// Compresses a supervised model with product quantization, mirroring fastText's
    /// <c>quantize</c>. <paramref name="cutoff"/> optionally prunes the vocabulary to the
    /// highest-norm embeddings first. Retraining after pruning is not yet supported.
    /// </summary>
    public void Quantize(int dsub = 2, bool qnorm = false, int cutoff = 0, bool qout = false, bool retrain = false)
    {
        if (_args.Model != ModelName.Sup)
        {
            throw new InvalidOperationException("For now we only support quantization of supervised models.");
        }
        if (_quantized)
        {
            throw new InvalidOperationException("Model is already quantized.");
        }
        if (_input is not DenseMatrix input || _output is not DenseMatrix output)
        {
            throw new InvalidOperationException("Quantization requires a dense (non-quantized) model.");
        }

        _args.Qout = qout;
        int dim = _args.Dim;

        if (cutoff > 0 && cutoff < input.Rows)
        {
            if (retrain)
            {
                throw new NotSupportedException("Retraining during quantization is not yet supported.");
            }
            List<int> idx = SelectEmbeddings(input, cutoff);
            _dict.Prune(idx);
            var ninput = new DenseMatrix(idx.Count, dim);
            for (int i = 0; i < idx.Count; i++)
            {
                Array.Copy(input.RawData, (long)idx[i] * dim, ninput.RawData, (long)i * dim, dim);
            }
            input = ninput;
        }

        _input = new QuantMatrix(input, dsub, qnorm);
        if (_args.Qout)
        {
            _output = new QuantMatrix(output, 2, qnorm);
        }
        _quantized = true;
        _precomputedWordVectors = null;
        _model = ModelLoader.BuildModel(_args, _dict, _input, _output);
    }

    private List<int> SelectEmbeddings(DenseMatrix input, int cutoff)
    {
        var norms = new float[input.Rows];
        input.L2NormRow(norms);
        var idx = new List<int>((int)input.Rows);
        for (int i = 0; i < input.Rows; i++)
        {
            idx.Add(i);
        }
        int eosid = _dict.EosId;
        idx.Sort((a, b) =>
        {
            bool aBeforeB = eosid == a || (eosid != b && norms[a] > norms[b]);
            if (aBeforeB)
            {
                return -1;
            }
            bool bBeforeA = eosid == b || (eosid != a && norms[b] > norms[a]);
            return bBeforeA ? 1 : 0;
        });
        idx.RemoveRange(cutoff, idx.Count - cutoff);
        return idx;
    }

    /// <summary>The words in the model dictionary, in id order.</summary>
    public IReadOnlyList<string> GetWords()
    {
        var words = new string[_dict.NWords];
        for (int i = 0; i < words.Length; i++)
        {
            words[i] = _dict.GetWord(i);
        }
        return words;
    }

    /// <summary>The labels in the model dictionary, in id order.</summary>
    public IReadOnlyList<string> GetLabels()
    {
        var labels = new string[_dict.NLabels];
        for (int i = 0; i < labels.Length; i++)
        {
            labels[i] = _dict.GetLabel(i);
        }
        return labels;
    }

    /// <summary>Dictionary id of a word, or -1 if it is not a known word.</summary>
    public int GetWordId(string word)
    {
        ArgumentNullException.ThrowIfNull(word);
        int id = _dict.GetId(Encoding.UTF8.GetBytes(word));
        if (id < 0 || _dict.GetEntryType(id) != EntryType.Word)
        {
            return -1;
        }
        return id;
    }

    /// <summary>Input-matrix row id of a subword (its hashed bucket).</summary>
    public int GetSubwordId(string subword)
    {
        ArgumentNullException.ThrowIfNull(subword);
        return _dict.GetSubwordId(Encoding.UTF8.GetBytes(subword));
    }

    /// <summary>Computes the embedding for a single word (averaging its subword n-grams).</summary>
    public float[] GetWordVector(string word)
    {
        ArgumentNullException.ThrowIfNull(word);
        var vec = new Vector(_args.Dim);
        AddWordVector(vec, Encoding.UTF8.GetBytes(word));
        return vec.Data.ToArray();
    }

    /// <summary>Computes the input-matrix vector for a single subword.</summary>
    public float[] GetSubwordVector(string subword)
    {
        ArgumentNullException.ThrowIfNull(subword);
        var vec = new Vector(_args.Dim);
        vec.AddRow(_input, _dict.GetSubwordId(Encoding.UTF8.GetBytes(subword)));
        return vec.Data.ToArray();
    }

    /// <summary>Computes a sentence/paragraph embedding by averaging word vectors.</summary>
    public float[] GetSentenceVector(string text)
    {
        ArgumentNullException.ThrowIfNull(text);
        var svec = new Vector(_args.Dim);
        svec.Zero();

        if (_args.Model == ModelName.Sup)
        {
            var line = new List<int>();
            _dict.GetLine(Encoding.UTF8.GetBytes(text), line);
            foreach (int id in line)
            {
                svec.AddRow(_input, id);
            }
            if (line.Count > 0)
            {
                svec.Mul(1f / line.Count);
            }
            return svec.Data.ToArray();
        }

        int count = 0;
        var word = new Vector(_args.Dim);
        foreach (string token in Tokenize(text))
        {
            AddWordVector(word, Encoding.UTF8.GetBytes(token));
            float norm = word.Norm();
            if (norm > 0)
            {
                word.Mul(1f / norm);
                svec.AddVector(word);
                count++;
            }
        }
        if (count > 0)
        {
            svec.Mul(1f / count);
        }
        return svec.Data.ToArray();
    }

    /// <summary>Returns each subword n-gram of a word together with its input vector.</summary>
    public IReadOnlyList<FastTextNgramVector> GetNgramVectors(string word)
    {
        ArgumentNullException.ThrowIfNull(word);
        var ngrams = new List<int>();
        var substrings = new List<string>();
        _dict.GetSubwords(Encoding.UTF8.GetBytes(word), ngrams, substrings);

        int count = Math.Min(ngrams.Count, substrings.Count);
        var result = new List<FastTextNgramVector>(count);
        for (int i = 0; i < count; i++)
        {
            var vec = new Vector(_args.Dim);
            if (ngrams[i] >= 0)
            {
                vec.AddRow(_input, ngrams[i]);
            }
            result.Add(new FastTextNgramVector(substrings[i], vec.Data.ToArray()));
        }
        return result;
    }

    /// <summary>Finds the <paramref name="k"/> words most similar to <paramref name="word"/>.</summary>
    public IReadOnlyList<FastTextNeighbor> GetNearestNeighbors(string word, int k = 10)
    {
        ArgumentNullException.ThrowIfNull(word);
        if (k <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(k), "k needs to be 1 or higher!");
        }

        var query = new Vector(_args.Dim);
        AddWordVector(query, Encoding.UTF8.GetBytes(word));

        var banSet = new HashSet<string>(StringComparer.Ordinal) { word };
        return FindNearest(query, k, banSet);
    }

    /// <summary>Solves the analogy "<paramref name="a"/> is to <paramref name="b"/> as <paramref name="c"/> is to ?".</summary>
    public IReadOnlyList<FastTextNeighbor> GetAnalogies(string a, string b, string c, int k = 10)
    {
        ArgumentNullException.ThrowIfNull(a);
        ArgumentNullException.ThrowIfNull(b);
        ArgumentNullException.ThrowIfNull(c);
        if (k <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(k), "k needs to be 1 or higher!");
        }

        var query = new Vector(_args.Dim);
        query.Zero();
        var buffer = new Vector(_args.Dim);

        AddWordVector(buffer, Encoding.UTF8.GetBytes(a));
        query.AddVector(buffer, 1f / (buffer.Norm() + SimilarityEpsilon));

        AddWordVector(buffer, Encoding.UTF8.GetBytes(b));
        query.AddVector(buffer, -1f / (buffer.Norm() + SimilarityEpsilon));

        AddWordVector(buffer, Encoding.UTF8.GetBytes(c));
        query.AddVector(buffer, 1f / (buffer.Norm() + SimilarityEpsilon));

        var banSet = new HashSet<string>(StringComparer.Ordinal) { a, b, c };
        return FindNearest(query, k, banSet);
    }

    /// <summary>Predicts the most likely labels for a line of text (supervised models only).</summary>
    public IReadOnlyList<FastTextPrediction> Predict(string text, int k = 1, float threshold = 0f)
    {
        ArgumentNullException.ThrowIfNull(text);
        if (k <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(k), "k needs to be 1 or higher!");
        }
        if (_args.Model != ModelName.Sup)
        {
            throw new InvalidOperationException("Model needs to be supervised for prediction!");
        }

        var words = new List<int>();
        _dict.GetLine(Encoding.UTF8.GetBytes(text), words);
        if (words.Count == 0)
        {
            return Array.Empty<FastTextPrediction>();
        }

        var heap = new List<Prediction>();
        var state = new ModelState(_args.Dim, _model.OutputSize, 0);
        _model.Predict(words, k, threshold, heap, state);

        var result = new FastTextPrediction[heap.Count];
        for (int i = 0; i < heap.Count; i++)
        {
            result[i] = new FastTextPrediction(
                _dict.GetLabel(heap[i].Label),
                (float)Math.Exp(heap[i].Score));
        }
        return result;
    }

    /// <summary>
    /// Evaluates the model over a set of labeled test lines, returning precision/recall metrics.
    /// Each line must be in fastText format (label tokens prefixed with the model's label prefix).
    /// </summary>
    public Meter Test(IEnumerable<string> lines, int k = 1, float threshold = 0f)
    {
        ArgumentNullException.ThrowIfNull(lines);
        if (k <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(k), "k needs to be 1 or higher!");
        }
        if (_args.Model != ModelName.Sup)
        {
            throw new InvalidOperationException("Model needs to be supervised for prediction!");
        }

        var meter = new Meter(falseNegativeLabels: false);
        var words = new List<int>();
        var labels = new List<int>();
        var heap = new List<Prediction>();
        var state = new ModelState(_args.Dim, _model.OutputSize, 0);

        foreach (string line in lines)
        {
            if (line is null)
            {
                continue;
            }
            _dict.GetLine(Encoding.UTF8.GetBytes(line), words, labels);
            if (labels.Count == 0 || words.Count == 0)
            {
                continue;
            }
            heap.Clear();
            _model.Predict(words, k, threshold, heap, state);
            meter.Log(labels, heap);
        }
        return meter;
    }

    /// <summary>Reads test lines from a stream and evaluates the model (see <see cref="Test(IEnumerable{string}, int, float)"/>).</summary>
    public Meter Test(Stream stream, int k = 1, float threshold = 0f)
    {
        ArgumentNullException.ThrowIfNull(stream);
        return Test(ReadLines(stream), k, threshold);
    }

    private static IEnumerable<string> ReadLines(Stream stream)
    {
        using var reader = new StreamReader(stream, Encoding.UTF8, detectEncodingFromByteOrderMarks: true, leaveOpen: true);
        string? line;
        while ((line = reader.ReadLine()) is not null)
        {
            yield return line;
        }
    }

    private void AddWordVector(Vector vec, ReadOnlySpan<byte> word)
    {
        List<int> ngrams = _dict.GetSubwords(word);
        vec.Zero();
        foreach (int id in ngrams)
        {
            vec.AddRow(_input, id);
        }
        if (ngrams.Count > 0)
        {
            vec.Mul(1f / ngrams.Count);
        }
    }

    private DenseMatrix PrecomputeWordVectors()
    {
        if (_precomputedWordVectors is not null)
        {
            return _precomputedWordVectors;
        }

        int nwords = _dict.NWords;
        var wordVectors = new DenseMatrix(nwords, _args.Dim);
        wordVectors.Zero();
        var vec = new Vector(_args.Dim);
        for (int i = 0; i < nwords; i++)
        {
            AddWordVector(vec, _dict.GetWordBytes(i));
            float norm = vec.Norm();
            if (norm > 0)
            {
                wordVectors.AddVectorToRow(vec.Data, i, 1f / norm);
            }
        }
        _precomputedWordVectors = wordVectors;
        return wordVectors;
    }

    private IReadOnlyList<FastTextNeighbor> FindNearest(Vector query, int k, HashSet<string> banSet)
    {
        DenseMatrix wordVectors = PrecomputeWordVectors();

        float queryNorm = query.Norm();
        if (Math.Abs(queryNorm) < SimilarityEpsilon)
        {
            queryNorm = 1f;
        }

        var results = new List<FastTextNeighbor>(_dict.NWords);
        for (int i = 0; i < _dict.NWords; i++)
        {
            string word = _dict.GetWord(i);
            if (banSet.Contains(word))
            {
                continue;
            }
            float dp = wordVectors.DotRow(query.Data, i);
            results.Add(new FastTextNeighbor(word, dp / queryNorm));
        }

        results.Sort(static (x, y) =>
        {
            int cmp = y.Similarity.CompareTo(x.Similarity);
            return cmp != 0 ? cmp : string.CompareOrdinal(x.Word, y.Word);
        });

        if (results.Count > k)
        {
            results.RemoveRange(k, results.Count - k);
        }
        return results;
    }

    private static IEnumerable<string> Tokenize(string text)
    {
        int start = -1;
        for (int i = 0; i < text.Length; i++)
        {
            char ch = text[i];
            bool ws = ch is ' ' or '\t' or '\n' or '\r' or '\v' or '\f';
            if (ws)
            {
                if (start >= 0)
                {
                    yield return text.Substring(start, i - start);
                    start = -1;
                }
            }
            else if (start < 0)
            {
                start = i;
            }
        }
        if (start >= 0)
        {
            yield return text.Substring(start);
        }
    }
}
