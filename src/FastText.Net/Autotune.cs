// Ported from Facebook fastText (src/autotune.cc, src/autotune.h).
// See THIRD-PARTY-NOTICES.md. Original: Copyright (c) Facebook, Inc. (MIT License).

using System.Diagnostics;

namespace FastTextNet;

/// <summary>Metric optimized by autotune. Mirrors fastText's <c>metric_name</c>.</summary>
public enum AutotuneMetric
{
    /// <summary>F1 score over all labels (default).</summary>
    F1Score,
    /// <summary>F1 score for a single label (set <see cref="FastTextAutotuneArgs.MetricLabel"/>).</summary>
    F1ScoreLabel,
    /// <summary>Best precision at or above a target recall (set <see cref="FastTextAutotuneArgs.MetricValue"/>).</summary>
    PrecisionAtRecall,
    /// <summary>Best precision at a target recall, for a single label.</summary>
    PrecisionAtRecallLabel,
    /// <summary>Best recall at or above a target precision (set <see cref="FastTextAutotuneArgs.MetricValue"/>).</summary>
    RecallAtPrecision,
    /// <summary>Best recall at a target precision, for a single label.</summary>
    RecallAtPrecisionLabel,
}

/// <summary>
/// Autotune options. Mirrors fastText's <c>-autotune-*</c> command-line flags. Autotune searches
/// the hyper-parameter space (epoch, lr, dim, wordNgrams, minn/maxn, bucket, dsub, loss) to
/// maximize a validation metric, optionally under a model-size budget.
/// </summary>
public sealed class FastTextAutotuneArgs
{
    /// <summary>Path to the labeled validation file used to score each candidate.</summary>
    public required string ValidationFile { get; set; }

    /// <summary>Search budget in seconds (fastText default 300).</summary>
    public int Duration { get; set; } = 300;

    /// <summary>Target model size in bytes; 0 means unlimited (no quantization).</summary>
    public long ModelSize { get; set; }

    /// <summary>Metric to maximize.</summary>
    public AutotuneMetric Metric { get; set; } = AutotuneMetric.F1Score;

    /// <summary>Label for the per-label metrics; required for the <c>*Label</c> variants.</summary>
    public string? MetricLabel { get; set; }

    /// <summary>Target recall/precision for the precision@recall / recall@precision metrics.</summary>
    public double MetricValue { get; set; }

    /// <summary>Number of predictions evaluated per example (the <c>k</c> in <c>predict</c>).</summary>
    public int Predictions { get; set; } = 1;

    /// <summary>Random seed for the search.</summary>
    public int Seed { get; set; }

    /// <summary>
    /// Optional cap on the number of trials. Independent of <see cref="Duration"/>; useful for
    /// reproducible, bounded runs (e.g. tests). Null means unbounded (time-limited only).
    /// </summary>
    public int? MaxTrials { get; set; }
}

public sealed partial class FastText
{
    /// <summary>
    /// Trains a supervised model, automatically searching for good hyper-parameters against a
    /// validation set. Mirrors fastText's <c>train_supervised(..., autotuneValidationFile=...)</c>.
    /// </summary>
    public static FastText TrainSupervisedAutotune(
        string inputPath, FastTextAutotuneArgs autotuneArgs, FastTextTrainArgs? baseArgs = null)
    {
        ArgumentNullException.ThrowIfNull(inputPath);
        ArgumentNullException.ThrowIfNull(autotuneArgs);
        if (!File.Exists(autotuneArgs.ValidationFile))
        {
            throw new FileNotFoundException("Validation file cannot be opened!", autotuneArgs.ValidationFile);
        }

        FastTextTrainArgs seed = (baseArgs ?? FastTextTrainArgs.ForSupervised()).Clone();
        seed.Model = FastTextArchitecture.Supervised;

        string[] validation = File.ReadAllLines(autotuneArgs.ValidationFile);
        var strategy = new AutotuneStrategy(seed, autotuneArgs.Seed);

        FastTextTrainArgs bestArgs = seed;
        int bestDsub = 2;
        double bestScore = double.NaN;
        int trials = 0;
        int sizeConstraintFailed = 0;

        var stopwatch = Stopwatch.StartNew();
        while (KeepTuning(stopwatch, autotuneArgs, trials))
        {
            trials++;
            double t = Progress(stopwatch, autotuneArgs, trials);
            (FastTextTrainArgs candidate, int dsub) = strategy.Ask(t);

            try
            {
                FastText model = Train(inputPath, candidate);
                bool sizeOk = autotuneArgs.ModelSize == 0
                    || model.TryQuantizeForSize(dsub, autotuneArgs.ModelSize);
                if (!sizeOk)
                {
                    sizeConstraintFailed++;
                    continue;
                }

                Meter meter = model.Test(validation, autotuneArgs.Predictions);
                double score = GetMetricScore(model, meter, autotuneArgs);

                if (double.IsNaN(bestScore) || score > bestScore)
                {
                    bestScore = score;
                    bestArgs = candidate;
                    bestDsub = dsub;
                    strategy.UpdateBest(candidate, dsub);
                }
            }
            catch (OutOfMemoryException)
            {
                // Ignore candidates whose parameters demand too much memory.
            }
        }

        if (double.IsNaN(bestScore))
        {
            string reason = sizeConstraintFailed > 0
                ? "Couldn't fulfil model size constraint: please increase the model size budget."
                : "Didn't have enough time to train once: please increase the autotune duration.";
            throw new InvalidOperationException(reason);
        }

        FastText best = Train(inputPath, bestArgs);
        if (autotuneArgs.ModelSize != 0)
        {
            best.TryQuantizeForSize(bestDsub, autotuneArgs.ModelSize);
        }
        return best;
    }

    private static bool KeepTuning(Stopwatch stopwatch, FastTextAutotuneArgs args, int trials)
    {
        if (args.MaxTrials is int max && trials >= max)
        {
            return false;
        }
        return stopwatch.Elapsed.TotalSeconds < args.Duration;
    }

    private static double Progress(Stopwatch stopwatch, FastTextAutotuneArgs args, int trials)
    {
        double byTime = args.Duration > 0 ? stopwatch.Elapsed.TotalSeconds / args.Duration : 0.0;
        double byTrials = args.MaxTrials is int max && max > 0 ? (double)trials / max : 0.0;
        return Math.Min(1.0, Math.Max(byTime, byTrials));
    }

    private static double GetMetricScore(FastText model, Meter meter, FastTextAutotuneArgs args)
    {
        int labelId = -1;
        if (args.Metric is AutotuneMetric.F1ScoreLabel
            or AutotuneMetric.PrecisionAtRecallLabel
            or AutotuneMetric.RecallAtPrecisionLabel)
        {
            if (string.IsNullOrEmpty(args.MetricLabel))
            {
                throw new InvalidOperationException("This autotune metric requires a metric label.");
            }
            labelId = model.GetLabelId(args.MetricLabel);
            if (labelId == -1)
            {
                throw new InvalidOperationException("Unknown autotune metric label.");
            }
        }

        return args.Metric switch
        {
            AutotuneMetric.F1Score => meter.F1Score(),
            AutotuneMetric.F1ScoreLabel => meter.F1Score(labelId),
            AutotuneMetric.PrecisionAtRecall => meter.PrecisionAtRecall(args.MetricValue),
            AutotuneMetric.PrecisionAtRecallLabel => meter.PrecisionAtRecall(labelId, args.MetricValue),
            AutotuneMetric.RecallAtPrecision => meter.RecallAtPrecision(args.MetricValue),
            AutotuneMetric.RecallAtPrecisionLabel => meter.RecallAtPrecision(labelId, args.MetricValue),
            _ => throw new InvalidOperationException("Unknown metric"),
        };
    }
}

// Random-search strategy: perturbs the best-known hyper-parameters with Gaussian noise whose
// spread shrinks as the search progresses. Mirrors fastText's AutotuneStrategy.
internal sealed class AutotuneStrategy
{
    private static readonly int[] MinnChoices = { 0, 2, 3 };

    private MinstdRand _rng;
    private FastTextTrainArgs _best = null!;
    private int _trials;
    private int _bestMinnIndex;
    private int _bestDsubExponent = 1;
    private int _bestNonzeroBucket = 2_000_000;
    private readonly int _originalBucket;

    public AutotuneStrategy(FastTextTrainArgs originalArgs, int seed)
    {
        _rng = new MinstdRand(seed);
        _originalBucket = originalArgs.Bucket;
        UpdateBest(originalArgs, 2);
    }

    public (FastTextTrainArgs Args, int Dsub) Ask(double t)
    {
        t = Math.Min(1.0, t);
        _trials++;

        if (_trials == 1)
        {
            return (_best.Clone(), 1 << _bestDsubExponent);
        }

        FastTextTrainArgs args = _best.Clone();
        int dsub = 1 << _bestDsubExponent;

        if (!args.IsManual("epoch"))
        {
            args.Epoch = UpdateGauss(args.Epoch, 1, 100, 2.8, 2.5, t, linear: false);
        }
        if (!args.IsManual("lr"))
        {
            args.Lr = UpdateGauss(args.Lr, 0.01, 5.0, 1.9, 1.0, t, linear: false);
        }
        if (!args.IsManual("dim"))
        {
            args.Dim = UpdateGauss(args.Dim, 1, 1000, 1.4, 0.3, t, linear: false);
        }
        if (!args.IsManual("wordNgrams"))
        {
            args.WordNgrams = UpdateGauss(args.WordNgrams, 1, 5, 4.3, 2.4, t, linear: true);
        }
        if (!args.IsManual("dsub"))
        {
            int dsubExponent = UpdateGauss(_bestDsubExponent, 1, 4, 2.0, 1.0, t, linear: true);
            dsub = 1 << dsubExponent;
        }
        if (!args.IsManual("minn"))
        {
            int minnIndex = UpdateGauss(_bestMinnIndex, 0, MinnChoices.Length - 1, 4.0, 1.4, t, linear: true);
            args.Minn = MinnChoices[minnIndex];
        }
        if (!args.IsManual("maxn"))
        {
            args.Maxn = args.Minn == 0 ? 0 : args.Minn + 3;
        }
        if (!args.IsManual("bucket"))
        {
            args.Bucket = UpdateGauss(_bestNonzeroBucket, 10000, 10000000, 2.0, 1.5, t, linear: false);
        }
        else
        {
            args.Bucket = _originalBucket;
        }
        if (args.WordNgrams <= 1 && args.Maxn == 0)
        {
            args.Bucket = 0;
        }
        if (!args.IsManual("loss"))
        {
            args.Loss = FastTextLossFunction.Softmax;
        }

        return (args, dsub);
    }

    public void UpdateBest(FastTextTrainArgs args, int dsub)
    {
        _best = args.Clone();
        _bestMinnIndex = Math.Max(0, Array.IndexOf(MinnChoices, args.Minn));
        _bestDsubExponent = (int)Math.Log2(dsub);
        if (args.Bucket != 0)
        {
            _bestNonzeroBucket = args.Bucket;
        }
    }

    private double GetGauss(double val, double startSigma, double endSigma, double t, bool linear)
    {
        double stddev = startSigma - ((startSigma - endSigma) / 0.5) * Math.Min(0.5, Math.Max(t - 0.25, 0.0));
        double coeff = _rng.NextGaussian(0.0, stddev);
        return linear ? coeff + val : Math.Pow(2.0, coeff) * val;
    }

    private int UpdateGauss(int val, int min, int max, double startSigma, double endSigma, double t, bool linear)
    {
        int result = (int)GetGauss(val, startSigma, endSigma, t, linear);
        return Math.Clamp(result, min, max);
    }

    private double UpdateGauss(double val, double min, double max, double startSigma, double endSigma, double t, bool linear)
    {
        double result = GetGauss(val, startSigma, endSigma, t, linear);
        return Math.Clamp(result, min, max);
    }
}
