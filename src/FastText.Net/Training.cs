// Ported from Facebook fastText (src/fasttext.cc train/supervised/cbow/skipgram).
// See THIRD-PARTY-NOTICES.md. Original: Copyright (c) Facebook, Inc. (MIT License).

namespace FastTextNet;

/// <summary>Model architecture to train.</summary>
public enum FastTextArchitecture
{
    /// <summary>Continuous bag-of-words unsupervised embeddings.</summary>
    Cbow,
    /// <summary>Skip-gram unsupervised embeddings.</summary>
    Skipgram,
    /// <summary>Supervised text classification.</summary>
    Supervised,
}

/// <summary>Loss function used during training.</summary>
public enum FastTextLossFunction
{
    /// <summary>Hierarchical softmax.</summary>
    HierarchicalSoftmax,
    /// <summary>Negative sampling.</summary>
    NegativeSampling,
    /// <summary>Full softmax.</summary>
    Softmax,
    /// <summary>One-vs-all (independent binary classifiers), for multi-label classification.</summary>
    OneVsAll,
}

/// <summary>
/// Training hyper-parameters, mirroring fastText's <c>Args</c>. Use <see cref="ForSupervised"/>,
/// <see cref="ForSkipgram"/>, or <see cref="ForCbow"/> for the matching default presets.
/// </summary>
public sealed class FastTextTrainArgs
{
    /// <summary>Model architecture to train.</summary>
    public FastTextArchitecture Model { get; set; } = FastTextArchitecture.Skipgram;
    /// <summary>Loss function.</summary>
    public FastTextLossFunction Loss { get; set; } = FastTextLossFunction.NegativeSampling;
    /// <summary>Size of word vectors.</summary>
    public int Dim { get; set; } = 100;
    /// <summary>Learning rate.</summary>
    public double Lr { get; set; } = 0.05;
    /// <summary>Rate of updates for the learning rate.</summary>
    public int LrUpdateRate { get; set; } = 100;
    /// <summary>Size of the context window.</summary>
    public int Ws { get; set; } = 5;
    /// <summary>Number of training epochs.</summary>
    public int Epoch { get; set; } = 5;
    /// <summary>Minimal number of word occurrences.</summary>
    public int MinCount { get; set; } = 5;
    /// <summary>Minimal number of label occurrences.</summary>
    public int MinCountLabel { get; set; }
    /// <summary>Number of negatives sampled (negative-sampling loss).</summary>
    public int Neg { get; set; } = 5;
    /// <summary>Maximum length of word n-grams.</summary>
    public int WordNgrams { get; set; } = 1;
    /// <summary>Number of buckets for hashing subwords and word n-grams.</summary>
    public int Bucket { get; set; } = 2000000;
    /// <summary>Minimum length of character n-grams.</summary>
    public int Minn { get; set; } = 3;
    /// <summary>Maximum length of character n-grams.</summary>
    public int Maxn { get; set; } = 6;
    /// <summary>Sampling threshold for frequent words.</summary>
    public double T { get; set; } = 1e-4;
    /// <summary>Random seed.</summary>
    public int Seed { get; set; }
    /// <summary>Number of blocks used to seed the random matrix initialization.</summary>
    public int Thread { get; set; } = 12;
    /// <summary>Label prefix that distinguishes labels from words in the corpus.</summary>
    public string Label { get; set; } = "__label__";

    /// <summary>Defaults matching fastText's <c>train_supervised</c>.</summary>
    public static FastTextTrainArgs ForSupervised() => new()
    {
        Model = FastTextArchitecture.Supervised,
        Loss = FastTextLossFunction.Softmax,
        Lr = 0.1,
        MinCount = 1,
        Minn = 0,
        Maxn = 0,
    };

    /// <summary>Defaults matching fastText's <c>train_unsupervised(model="skipgram")</c>.</summary>
    public static FastTextTrainArgs ForSkipgram() => new()
    {
        Model = FastTextArchitecture.Skipgram,
        Loss = FastTextLossFunction.NegativeSampling,
    };

    /// <summary>Defaults matching fastText's <c>train_unsupervised(model="cbow")</c>.</summary>
    public static FastTextTrainArgs ForCbow() => new()
    {
        Model = FastTextArchitecture.Cbow,
        Loss = FastTextLossFunction.NegativeSampling,
    };

    internal Args ToInternal() => new()
    {
        Model = Model switch
        {
            FastTextArchitecture.Cbow => ModelName.Cbow,
            FastTextArchitecture.Skipgram => ModelName.Sg,
            FastTextArchitecture.Supervised => ModelName.Sup,
            _ => throw new ArgumentOutOfRangeException(nameof(Model)),
        },
        Loss = Loss switch
        {
            FastTextLossFunction.HierarchicalSoftmax => LossName.Hs,
            FastTextLossFunction.NegativeSampling => LossName.Ns,
            FastTextLossFunction.Softmax => LossName.Softmax,
            FastTextLossFunction.OneVsAll => LossName.Ova,
            _ => throw new ArgumentOutOfRangeException(nameof(Loss)),
        },
        Dim = Dim,
        Lr = Lr,
        LrUpdateRate = LrUpdateRate,
        Ws = Ws,
        Epoch = Epoch,
        MinCount = MinCount,
        MinCountLabel = MinCountLabel,
        Neg = Neg,
        WordNgrams = WordNgrams,
        Bucket = Bucket,
        Minn = Minn,
        Maxn = Maxn,
        T = T,
        Seed = Seed,
        Thread = Thread,
        Label = Label,
    };
}

public sealed partial class FastText
{
    /// <summary>Trains a supervised classification model from a labeled corpus file.</summary>
    public static FastText TrainSupervised(string inputPath, FastTextTrainArgs? args = null) =>
        Train(inputPath, args ?? FastTextTrainArgs.ForSupervised());

    /// <summary>Trains unsupervised word embeddings (skip-gram by default) from a corpus file.</summary>
    public static FastText TrainUnsupervised(string inputPath, FastTextTrainArgs? args = null) =>
        Train(inputPath, args ?? FastTextTrainArgs.ForSkipgram());

    /// <summary>Trains a model from a corpus file using the given hyper-parameters.</summary>
    public static FastText Train(string inputPath, FastTextTrainArgs trainArgs)
    {
        ArgumentNullException.ThrowIfNull(inputPath);
        ArgumentNullException.ThrowIfNull(trainArgs);

        Args args = trainArgs.ToInternal();
        byte[] data = File.ReadAllBytes(inputPath);

        var dict = new Dictionary(args);
        dict.ReadFromFile(data);

        var input = new DenseMatrix(dict.NWords + args.Bucket, args.Dim);
        input.Uniform(1.0f / args.Dim, args.Thread, args.Seed);

        int outputRows = args.Model == ModelName.Sup ? dict.NLabels : dict.NWords;
        var output = new DenseMatrix(outputRows, args.Dim);
        output.Zero();

        Loss loss = ModelLoader.CreateLoss(args, dict, output);
        var model = new Model(input, output, loss, normalizeGradient: args.Model == ModelName.Sup);

        TrainLoop(args, dict, model, data);

        return new FastText(args, dict, input, output, model);
    }

    private static void TrainLoop(Args args, Dictionary dict, Model model, byte[] data)
    {
        List<(int Start, int Length)> lines = SplitLines(data);
        long ntokens = dict.NTokens;
        long total = (long)args.Epoch * ntokens;
        int outputRows = args.Model == ModelName.Sup ? dict.NLabels : dict.NWords;

        var state = new ModelState(args.Dim, outputRows, args.Seed);
        var line = new List<int>();
        var labels = new List<int>();
        var bow = new List<int>();

        long tokenCount = 0;
        while (tokenCount < total)
        {
            foreach ((int start, int length) in lines)
            {
                if (tokenCount >= total)
                {
                    break;
                }
                ReadOnlySpan<byte> span = data.AsSpan(start, length);
                float progress = (float)((double)tokenCount / total);
                float lr = (float)(args.Lr * (1.0 - progress));

                switch (args.Model)
                {
                    case ModelName.Sup:
                        tokenCount += dict.GetLine(span, line, labels);
                        Supervised(model, args, state, lr, line, labels);
                        break;
                    case ModelName.Cbow:
                        tokenCount += dict.GetLine(span, line, ref state.Rng);
                        Cbow(model, dict, args, state, lr, line, bow);
                        break;
                    case ModelName.Sg:
                        tokenCount += dict.GetLine(span, line, ref state.Rng);
                        Skipgram(model, dict, args, state, lr, line);
                        break;
                }
            }
        }
    }

    private static void Supervised(
        Model model, Args args, ModelState state, float lr, List<int> line, List<int> labels)
    {
        if (labels.Count == 0 || line.Count == 0)
        {
            return;
        }
        if (args.Loss == LossName.Ova)
        {
            model.Update(line, labels, Model.AllLabelsAsTarget, lr, state);
        }
        else
        {
            int i = state.Rng.NextInt(0, labels.Count - 1);
            model.Update(line, labels, i, lr, state);
        }
    }

    private static void Cbow(
        Model model, Dictionary dict, Args args, ModelState state, float lr, List<int> line, List<int> bow)
    {
        for (int w = 0; w < line.Count; w++)
        {
            int boundary = state.Rng.NextInt(1, args.Ws);
            bow.Clear();
            for (int c = -boundary; c <= boundary; c++)
            {
                if (c != 0 && w + c >= 0 && w + c < line.Count)
                {
                    bow.AddRange(dict.GetSubwordsById(line[w + c]));
                }
            }
            model.Update(bow, line, w, lr, state);
        }
    }

    private static void Skipgram(
        Model model, Dictionary dict, Args args, ModelState state, float lr, List<int> line)
    {
        for (int w = 0; w < line.Count; w++)
        {
            int boundary = state.Rng.NextInt(1, args.Ws);
            var ngrams = new List<int>(dict.GetSubwordsById(line[w]));
            for (int c = -boundary; c <= boundary; c++)
            {
                if (c != 0 && w + c >= 0 && w + c < line.Count)
                {
                    model.Update(ngrams, line, w + c, lr, state);
                }
            }
        }
    }

    private static List<(int Start, int Length)> SplitLines(byte[] data)
    {
        var lines = new List<(int, int)>();
        int start = 0;
        for (int i = 0; i < data.Length; i++)
        {
            if (data[i] == (byte)'\n')
            {
                lines.Add((start, i - start));
                start = i + 1;
            }
        }
        if (start < data.Length)
        {
            lines.Add((start, data.Length - start));
        }
        return lines;
    }
}
