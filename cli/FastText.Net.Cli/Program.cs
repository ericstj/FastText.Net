// A managed re-implementation of the fastText command-line tool (src/main.cc).
// See THIRD-PARTY-NOTICES.md. Original: Copyright (c) Facebook, Inc. (MIT License).

using System.Globalization;
using FastTextNet;

namespace FastTextNet.Cli;

internal static class Program
{
    private static readonly CultureInfo Inv = CultureInfo.InvariantCulture;

    private static int Main(string[] args)
    {
        if (args.Length < 1)
        {
            PrintUsage();
            return 1;
        }

        try
        {
            string command = args[0];
            switch (command)
            {
                case "supervised":
                case "skipgram":
                case "cbow":
                    return Train(command, args);
                case "test":
                case "test-label":
                    return Test(command, args);
                case "quantize":
                    return Quantize(args);
                case "predict":
                case "predict-prob":
                    return Predict(command, args);
                case "print-word-vectors":
                    return PrintWordVectors(args);
                case "print-sentence-vectors":
                    return PrintSentenceVectors(args);
                case "print-ngrams":
                    return PrintNgrams(args);
                case "nn":
                    return Nn(args);
                case "analogies":
                    return Analogies(args);
                case "dump":
                    return Dump(args);
                default:
                    PrintUsage();
                    return 1;
            }
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine(ex.Message);
            return 1;
        }
    }

    private static void PrintUsage()
    {
        Console.Error.WriteLine(
            """
            usage: fasttext <command> <args>

            The commands supported by fasttext are:

              supervised              train a supervised classifier
              quantize                quantize a model to reduce the memory usage
              test                    evaluate a supervised classifier
              test-label              print labels with precision and recall scores
              predict                 predict most likely labels
              predict-prob            predict most likely labels with probabilities
              skipgram                train a skipgram model
              cbow                    train a cbow model
              print-word-vectors      print word vectors given a trained model
              print-sentence-vectors  print sentence vectors given a trained model
              print-ngrams            print ngrams given a trained model and word
              nn                      query for nearest neighbors
              analogies               query for analogies
              dump                    dump arguments,dictionary,input/output vectors
            """);
    }

    private static int Train(string command, string[] args)
    {
        var p = new ArgParser(args);
        FastTextTrainArgs trainArgs = command switch
        {
            "supervised" => FastTextTrainArgs.ForSupervised(),
            "cbow" => FastTextTrainArgs.ForCbow(),
            _ => FastTextTrainArgs.ForSkipgram(),
        };

        string? input = null;
        string? output = null;
        bool saveOutput = false;
        string? autotuneValidation = null;
        string? autotuneMetric = null;
        int autotunePredictions = 1;
        int autotuneDuration = 300;
        string? autotuneModelSize = null;

        p.ForEach((flag, get) =>
        {
            switch (flag)
            {
                case "-input": input = get(); break;
                case "-output": output = get(); break;
                case "-lr": trainArgs.Lr = double.Parse(get(), Inv); trainArgs.PinnedArgs.Add("lr"); break;
                case "-lrUpdateRate": trainArgs.LrUpdateRate = int.Parse(get(), Inv); break;
                case "-dim": trainArgs.Dim = int.Parse(get(), Inv); trainArgs.PinnedArgs.Add("dim"); break;
                case "-ws": trainArgs.Ws = int.Parse(get(), Inv); break;
                case "-epoch": trainArgs.Epoch = int.Parse(get(), Inv); trainArgs.PinnedArgs.Add("epoch"); break;
                case "-minCount": trainArgs.MinCount = int.Parse(get(), Inv); break;
                case "-minCountLabel": trainArgs.MinCountLabel = int.Parse(get(), Inv); break;
                case "-neg": trainArgs.Neg = int.Parse(get(), Inv); break;
                case "-wordNgrams": trainArgs.WordNgrams = int.Parse(get(), Inv); trainArgs.PinnedArgs.Add("wordNgrams"); break;
                case "-loss": trainArgs.Loss = ParseLoss(get()); trainArgs.PinnedArgs.Add("loss"); break;
                case "-bucket": trainArgs.Bucket = int.Parse(get(), Inv); trainArgs.PinnedArgs.Add("bucket"); break;
                case "-minn": trainArgs.Minn = int.Parse(get(), Inv); trainArgs.PinnedArgs.Add("minn"); break;
                case "-maxn": trainArgs.Maxn = int.Parse(get(), Inv); trainArgs.PinnedArgs.Add("maxn"); break;
                case "-thread": trainArgs.Thread = int.Parse(get(), Inv); break;
                case "-t": trainArgs.T = double.Parse(get(), Inv); break;
                case "-label": trainArgs.Label = get(); break;
                case "-seed": trainArgs.Seed = int.Parse(get(), Inv); break;
                case "-saveOutput": saveOutput = true; break;
                case "-verbose": get(); break;
                case "-autotune-validation": autotuneValidation = get(); break;
                case "-autotune-metric": autotuneMetric = get(); break;
                case "-autotune-predictions": autotunePredictions = int.Parse(get(), Inv); break;
                case "-autotune-duration": autotuneDuration = int.Parse(get(), Inv); break;
                case "-autotune-modelsize": autotuneModelSize = get(); break;
                default: throw new ArgumentException($"Unknown argument: {flag}");
            }
        });

        if (string.IsNullOrEmpty(input) || string.IsNullOrEmpty(output))
        {
            throw new ArgumentException("Empty input or output path.");
        }

        bool hasAutotune = !string.IsNullOrEmpty(autotuneValidation);
        if (trainArgs.WordNgrams <= 1 && trainArgs.Maxn == 0 && !hasAutotune)
        {
            trainArgs.Bucket = 0;
        }

        FastText model;
        long modelSize = ParseModelSize(autotuneModelSize);
        if (hasAutotune)
        {
            (AutotuneMetric metric, string? label, double value) = ParseMetric(autotuneMetric);
            var at = new FastTextAutotuneArgs
            {
                ValidationFile = autotuneValidation!,
                Duration = autotuneDuration,
                ModelSize = modelSize,
                Metric = metric,
                MetricLabel = label,
                MetricValue = value,
                Predictions = autotunePredictions,
                Seed = trainArgs.Seed,
            };
            model = FastText.TrainSupervisedAutotune(input!, at, trainArgs);
        }
        else
        {
            model = FastText.Train(input!, trainArgs);
        }

        string modelFile = hasAutotune && modelSize != 0 ? output + ".ftz" : output + ".bin";
        model.SaveModel(modelFile);
        if (!model.IsQuantized)
        {
            model.SaveVectors(output + ".vec");
            if (saveOutput)
            {
                model.SaveOutput(output + ".output");
            }
        }
        return 0;
    }

    private static int Quantize(string[] args)
    {
        var p = new ArgParser(args);
        string? output = null;
        int dsub = 2;
        int cutoff = 0;
        bool qnorm = false;
        bool qout = false;
        bool retrain = false;

        p.ForEach((flag, get) =>
        {
            switch (flag)
            {
                case "-output": output = get(); break;
                case "-dsub": dsub = int.Parse(get(), Inv); break;
                case "-cutoff": cutoff = int.Parse(get(), Inv); break;
                case "-qnorm": qnorm = true; break;
                case "-qout": qout = true; break;
                case "-retrain": retrain = true; break;
                case "-input": get(); break;
                case "-verbose": get(); break;
                default: throw new ArgumentException($"Unknown argument: {flag}");
            }
        });

        if (string.IsNullOrEmpty(output))
        {
            throw new ArgumentException("Empty output path.");
        }

        FastText model = FastText.LoadModel(output + ".bin");
        model.Quantize(dsub, qnorm, cutoff, qout, retrain);
        model.SaveModel(output + ".ftz");
        return 0;
    }

    private static int Test(string command, string[] args)
    {
        bool perLabel = command == "test-label";
        if (args.Length < 3 || args.Length > 5)
        {
            Console.Error.WriteLine($"usage: fasttext {command} <model> <test-data> [<k>] [<th>]");
            return 1;
        }

        string modelPath = args[1];
        string input = args[2];
        int k = args.Length > 3 ? int.Parse(args[3], Inv) : 1;
        float threshold = args.Length > 4 ? float.Parse(args[4], Inv) : 0f;

        FastText model = FastText.LoadModel(modelPath);
        Meter meter = input == "-"
            ? model.Test(ReadLines(Console.In), k, threshold)
            : model.Test(File.ReadLines(input), k, threshold);

        if (perLabel)
        {
            IReadOnlyList<string> labels = model.GetLabels();
            for (int labelId = 0; labelId < labels.Count; labelId++)
            {
                WriteMetric("F1-Score", meter.F1Score(labelId));
                WriteMetric("Precision", meter.Precision(labelId));
                WriteMetric("Recall", meter.Recall(labelId));
                Console.WriteLine($" {labels[labelId]}");
            }
        }

        Console.WriteLine($"N\t{meter.NExamples}");
        Console.WriteLine($"P@{k}\t{meter.Precision().ToString("F3", Inv)}");
        Console.WriteLine($"R@{k}\t{meter.Recall().ToString("F3", Inv)}");
        return 0;
    }

    private static void WriteMetric(string name, double value)
    {
        string text = double.IsFinite(value) ? value.ToString("F6", Inv) : "--------";
        Console.Write($"{name} : {text}  ");
    }

    private static int Predict(string command, string[] args)
    {
        if (args.Length < 3 || args.Length > 5)
        {
            Console.Error.WriteLine($"usage: fasttext {command} <model> <test-data> [<k>] [<th>]");
            return 1;
        }

        bool printProb = command == "predict-prob";
        string modelPath = args[1];
        string input = args[2];
        int k = args.Length > 3 ? int.Parse(args[3], Inv) : 1;
        float threshold = args.Length > 4 ? float.Parse(args[4], Inv) : 0f;

        FastText model = FastText.LoadModel(modelPath);
        IEnumerable<string> lines = input == "-" ? ReadLines(Console.In) : File.ReadLines(input);

        foreach (string line in lines)
        {
            IReadOnlyList<FastTextPrediction> predictions = model.Predict(line, k, threshold);
            bool first = true;
            foreach (FastTextPrediction prediction in predictions)
            {
                if (!first)
                {
                    Console.Write(' ');
                }
                first = false;
                Console.Write(prediction.Label);
                if (printProb)
                {
                    Console.Write($" {prediction.Probability.ToString(Inv)}");
                }
            }
            Console.WriteLine();
        }
        return 0;
    }

    private static int PrintWordVectors(string[] args)
    {
        if (args.Length != 2)
        {
            Console.Error.WriteLine("usage: fasttext print-word-vectors <model>");
            return 1;
        }
        FastText model = FastText.LoadModel(args[1]);
        string? word;
        while ((word = ReadToken(Console.In)) is not null)
        {
            Console.WriteLine($"{word}{FormatVector(model.GetWordVector(word))}");
        }
        return 0;
    }

    private static int PrintSentenceVectors(string[] args)
    {
        if (args.Length != 2)
        {
            Console.Error.WriteLine("usage: fasttext print-sentence-vectors <model>");
            return 1;
        }
        FastText model = FastText.LoadModel(args[1]);
        foreach (string line in ReadLines(Console.In))
        {
            Console.WriteLine(FormatVector(model.GetSentenceVector(line)).TrimStart());
        }
        return 0;
    }

    private static int PrintNgrams(string[] args)
    {
        if (args.Length != 3)
        {
            Console.Error.WriteLine("usage: fasttext print-ngrams <model> <word>");
            return 1;
        }
        FastText model = FastText.LoadModel(args[1]);
        foreach (FastTextNgramVector ngram in model.GetNgramVectors(args[2]))
        {
            Console.WriteLine($"{ngram.Ngram}{FormatVector(ngram.Vector)}");
        }
        return 0;
    }

    private static int Nn(string[] args)
    {
        if (args.Length is < 2 or > 3)
        {
            Console.Error.WriteLine("usage: fasttext nn <model> <k>");
            Console.Error.WriteLine("  <model>      model filename");
            Console.Error.WriteLine("  <k>          (optional; 10 by default) predict top k labels");
            return 1;
        }
        int k = args.Length == 3 ? int.Parse(args[2], Inv) : 10;
        FastText model = FastText.LoadModel(args[1]);

        const string prompt = "Query word? ";
        Console.Write(prompt);
        string? word;
        while ((word = ReadToken(Console.In)) is not null)
        {
            foreach (FastTextNeighbor n in model.GetNearestNeighbors(word, k))
            {
                Console.WriteLine($"{n.Word} {n.Similarity.ToString(Inv)}");
            }
            Console.Write(prompt);
        }
        return 0;
    }

    private static int Analogies(string[] args)
    {
        if (args.Length is < 2 or > 3)
        {
            Console.Error.WriteLine("usage: fasttext analogies <model> <k>");
            Console.Error.WriteLine("  <model>      model filename");
            Console.Error.WriteLine("  <k>          (optional; 10 by default) predict top k labels");
            return 1;
        }
        int k = args.Length == 3 ? int.Parse(args[2], Inv) : 10;
        if (k <= 0)
        {
            throw new ArgumentException("k needs to be 1 or higher!");
        }
        FastText model = FastText.LoadModel(args[1]);

        const string prompt = "Query triplet (A - B + C)? ";
        Console.Write(prompt);
        while (true)
        {
            string? a = ReadToken(Console.In);
            string? b = ReadToken(Console.In);
            string? c = ReadToken(Console.In);
            if (a is null || b is null || c is null)
            {
                break;
            }
            foreach (FastTextNeighbor n in model.GetAnalogies(a, b, c, k))
            {
                Console.WriteLine($"{n.Word} {n.Similarity.ToString(Inv)}");
            }
            Console.Write(prompt);
        }
        return 0;
    }

    private static int Dump(string[] args)
    {
        if (args.Length < 3)
        {
            Console.Error.WriteLine("usage: fasttext dump <model> <option>");
            return 1;
        }
        FastText model = FastText.LoadModel(args[1]);
        FastText.DumpOption option = args[2] switch
        {
            "args" => FastText.DumpOption.Args,
            "dict" => FastText.DumpOption.Dict,
            "input" => FastText.DumpOption.Input,
            "output" => FastText.DumpOption.Output,
            _ => throw new ArgumentException("usage: fasttext dump <model> <option> (args,dict,input,output)"),
        };

        if ((option is FastText.DumpOption.Input or FastText.DumpOption.Output) && model.IsQuantized)
        {
            Console.Error.WriteLine("Not supported for quantized models.");
            return 0;
        }
        model.Dump(Console.Out, option);
        return 0;
    }

    private static FastTextLossFunction ParseLoss(string value) => value switch
    {
        "hs" => FastTextLossFunction.HierarchicalSoftmax,
        "ns" => FastTextLossFunction.NegativeSampling,
        "softmax" => FastTextLossFunction.Softmax,
        "ova" or "one-vs-all" => FastTextLossFunction.OneVsAll,
        _ => throw new ArgumentException($"Unknown loss: {value}"),
    };

    // Parses fastText's metric string, e.g. "f1", "f1:LABEL", "precisionAtRecall:30",
    // "recallAtPrecision:30:LABEL".
    private static (AutotuneMetric Metric, string? Label, double Value) ParseMetric(string? metric)
    {
        if (string.IsNullOrEmpty(metric) || metric == "f1")
        {
            return (AutotuneMetric.F1Score, null, 0.0);
        }
        if (metric.StartsWith("f1:", StringComparison.Ordinal))
        {
            return (AutotuneMetric.F1ScoreLabel, metric[3..], 0.0);
        }

        foreach ((string prefix, AutotuneMetric bare, AutotuneMetric labelled) in new[]
        {
            ("precisionAtRecall:", AutotuneMetric.PrecisionAtRecall, AutotuneMetric.PrecisionAtRecallLabel),
            ("recallAtPrecision:", AutotuneMetric.RecallAtPrecision, AutotuneMetric.RecallAtPrecisionLabel),
        })
        {
            if (!metric.StartsWith(prefix, StringComparison.Ordinal))
            {
                continue;
            }
            string rest = metric[prefix.Length..];
            int colon = rest.IndexOf(':');
            string valueStr = colon >= 0 ? rest[..colon] : rest;
            double value = double.Parse(valueStr, Inv) / 100.0;
            return colon >= 0
                ? (labelled, rest[(colon + 1)..], value)
                : (bare, null, value);
        }

        throw new ArgumentException($"Unknown metric : {metric}");
    }

    // Parses fastText's model-size string (e.g. "100M", "2G"). Empty means unlimited (0).
    private static long ParseModelSize(string? modelSize)
    {
        if (string.IsNullOrEmpty(modelSize))
        {
            return 0;
        }
        long multiplier = char.ToLowerInvariant(modelSize[^1]) switch
        {
            'k' => 1_000,
            'm' => 1_000_000,
            'g' => 1_000_000_000,
            _ => 1,
        };
        string digits = multiplier == 1 ? modelSize : modelSize[..^1];
        return long.Parse(digits, Inv) * multiplier;
    }

    private static string FormatVector(ReadOnlySpan<float> data)
    {
        var sb = new System.Text.StringBuilder();
        foreach (float value in data)
        {
            sb.Append(' ').Append(value.ToString("g6", Inv));
        }
        return sb.ToString();
    }

    private static IEnumerable<string> ReadLines(TextReader reader)
    {
        string? line;
        while ((line = reader.ReadLine()) is not null)
        {
            yield return line;
        }
    }

    private static string? ReadToken(TextReader reader)
    {
        int c;
        while ((c = reader.Read()) != -1 && char.IsWhiteSpace((char)c))
        {
            // skip leading whitespace
        }
        if (c == -1)
        {
            return null;
        }
        var sb = new System.Text.StringBuilder();
        do
        {
            sb.Append((char)c);
        }
        while ((c = reader.Read()) != -1 && !char.IsWhiteSpace((char)c));
        return sb.ToString();
    }
}

// Iterates fastText-style "-flag [value]" command-line arguments, starting after the command.
internal sealed class ArgParser
{
    private readonly string[] _args;

    public ArgParser(string[] args) => _args = args;

    public void ForEach(Action<string, Func<string>> handle)
    {
        for (int i = 1; i < _args.Length; i++)
        {
            string flag = _args[i];
            if (flag.Length == 0 || flag[0] != '-')
            {
                throw new ArgumentException("Provided argument without a dash!");
            }

            int valueIndex = i;
            string GetValue()
            {
                valueIndex = i + 1;
                if (valueIndex >= _args.Length)
                {
                    throw new ArgumentException($"{flag} is missing an argument");
                }
                return _args[valueIndex];
            }

            handle(flag, GetValue);
            i = valueIndex;
        }
    }
}
