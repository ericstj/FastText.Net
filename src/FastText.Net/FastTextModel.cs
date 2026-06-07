// Ported from Facebook fastText (src/fasttext.h, src/fasttext.cc).
// See THIRD-PARTY-NOTICES.md. Original: Copyright (c) Facebook, Inc. (MIT License).

using System.Buffers;
using System.Text;

namespace FastTextNet;

/// <summary>A single classification result.</summary>
public readonly record struct FastTextPrediction(string Label, float Probability);

/// <summary>
/// A managed, dependency-free fastText model for inference. Load a pretrained supervised
/// model (e.g. <c>lid.176.ftz</c> for language identification) and call
/// <see cref="Predict(string,int,float)"/>.
/// </summary>
/// <remarks>Instances are immutable after loading and safe to share across threads.</remarks>
public sealed class FastTextModel
{
    private const int MagicInt32 = 793712314;
    private const int SupportedVersion = 12;

    private readonly Args _args;
    private readonly Dictionary _dict;
    private readonly Model _model;
    private readonly bool _quantized;

    [ThreadStatic] private static List<int>? _wordsBuffer;
    [ThreadStatic] private static List<Prediction>? _heapBuffer;
    [ThreadStatic] private static ModelState? _stateBuffer;

    private FastTextModel(Args args, Dictionary dict, Model model, bool quantized)
    {
        _args = args;
        _dict = dict;
        _model = model;
        _quantized = quantized;
    }

    /// <summary>Embedding dimension of the model.</summary>
    public int Dimension => _args.Dim;

    /// <summary>Number of output labels.</summary>
    public int LabelCount => _dict.NLabels;

    /// <summary>True if the model uses product quantization (a <c>.ftz</c> model).</summary>
    public bool IsQuantized => _quantized;

    /// <summary>Loads a model from a file path.</summary>
    public static FastTextModel Load(string path)
    {
        using var stream = new FileStream(
            path, FileMode.Open, FileAccess.Read, FileShare.Read, 1 << 20, FileOptions.SequentialScan);
        return Load(stream);
    }

    /// <summary>Loads a model from a stream.</summary>
    public static FastTextModel Load(Stream stream)
    {
        using var reader = new BinaryReader(stream, Encoding.UTF8, leaveOpen: true);

        int magic = reader.ReadInt32();
        if (magic != MagicInt32)
        {
            throw new InvalidDataException("Not a fastText model file (bad magic number).");
        }
        int version = reader.ReadInt32();
        if (version > SupportedVersion)
        {
            throw new InvalidDataException(
                $"Unsupported fastText model version {version} (supported up to {SupportedVersion}).");
        }

        var args = new Args();
        args.Load(reader);
        if (version == 11 && args.Model == ModelName.Sup)
        {
            args.Maxn = 0;
        }

        var dict = new Dictionary(args);
        dict.Load(reader);

        bool quantInput = reader.ReadBoolean();
        Matrix input = quantInput ? new QuantMatrix() : new DenseMatrix();
        input.Load(reader);

        if (!quantInput && dict.IsPruned)
        {
            throw new InvalidDataException(
                "Invalid model file: pruned dictionary without quantized input. " +
                "Please download an updated model from fasttext.cc.");
        }

        args.Qout = reader.ReadBoolean();
        Matrix output = quantInput && args.Qout ? new QuantMatrix() : new DenseMatrix();
        output.Load(reader);

        Loss loss = CreateLoss(args, dict, output);
        var model = new Model(input, output, loss);
        return new FastTextModel(args, dict, model, quantInput);
    }

    private static Loss CreateLoss(Args args, Dictionary dict, Matrix output)
    {
        IReadOnlyList<long> Counts() => args.Model == ModelName.Sup
            ? dict.GetCounts(EntryType.Label)
            : dict.GetCounts(EntryType.Word);

        return args.Loss switch
        {
            LossName.Hs => new HierarchicalSoftmaxLoss(output, Counts()),
            LossName.Ns => new SigmoidLoss(output),
            LossName.Softmax => new SoftmaxLoss(output),
            LossName.Ova => new SigmoidLoss(output),
            _ => throw new InvalidDataException("Unknown loss function in model."),
        };
    }

    /// <summary>
    /// Predicts the most likely labels for a line of text.
    /// </summary>
    /// <param name="text">The input text (a single line; newlines are treated as separators).</param>
    /// <param name="k">Number of labels to return, ordered by descending probability.</param>
    /// <param name="threshold">Minimum probability for a label to be returned.</param>
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

        int byteCount = Encoding.UTF8.GetByteCount(text);
        byte[] bytes = ArrayPool<byte>.Shared.Rent(byteCount);
        try
        {
            Encoding.UTF8.GetBytes(text, bytes);
            return PredictBytes(bytes.AsSpan(0, byteCount), k, threshold);
        }
        finally
        {
            ArrayPool<byte>.Shared.Return(bytes);
        }
    }

    private IReadOnlyList<FastTextPrediction> PredictBytes(ReadOnlySpan<byte> bytes, int k, float threshold)
    {
        List<int> words = _wordsBuffer ??= new List<int>(256);
        List<Prediction> heap = _heapBuffer ??= new List<Prediction>();
        ModelState state = _stateBuffer ??= new ModelState(_args.Dim, _model.OutputSize);
        if (state.Output.Length != _model.OutputSize)
        {
            state = _stateBuffer = new ModelState(_args.Dim, _model.OutputSize);
        }

        _dict.GetLine(bytes, words);
        heap.Clear();
        if (words.Count == 0)
        {
            return Array.Empty<FastTextPrediction>();
        }

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
}
