// Ported from Facebook fastText (src/meter.h, src/meter.cc).
// See THIRD-PARTY-NOTICES.md. Original: Copyright (c) Facebook, Inc. (MIT License).

namespace FastTextNet;

/// <summary>Precision/recall/F1 metrics for a single label or the whole test set.</summary>
public readonly record struct FastTextMetrics(double Precision, double Recall, double F1Score);

/// <summary>
/// Accumulates prediction outcomes to compute precision, recall, and F1, mirroring
/// fastText's <c>Meter</c>.
/// </summary>
public sealed class Meter
{
    private sealed class LabelMetrics
    {
        public ulong Gold;
        public ulong Predicted;
        public ulong PredictedGold;
        public readonly List<(float Score, float Gold)> ScoreVsTrue = new();

        public double Precision() => Predicted == 0 ? double.NaN : PredictedGold / (double)Predicted;
        public double Recall() => Gold == 0 ? double.NaN : PredictedGold / (double)Gold;
        public double F1Score() =>
            Predicted + Gold == 0 ? double.NaN : 2.0 * PredictedGold / (Predicted + Gold);
    }

    private readonly LabelMetrics _metrics = new();
    private readonly Dictionary<int, LabelMetrics> _labelMetrics = new();
    private readonly bool _falseNegativeLabels;
    private ulong _nexamples;

    internal Meter(bool falseNegativeLabels) => _falseNegativeLabels = falseNegativeLabels;

    /// <summary>Number of examples logged.</summary>
    public long NExamples => (long)_nexamples;

    /// <summary>Records the gold labels and predictions for one example.</summary>
    internal void Log(IReadOnlyList<int> labels, IReadOnlyList<Prediction> predictions)
    {
        _nexamples++;
        _metrics.Gold += (ulong)labels.Count;
        _metrics.Predicted += (ulong)predictions.Count;

        foreach (Prediction prediction in predictions)
        {
            LabelMetrics lm = GetLabel(prediction.Label);
            lm.Predicted++;

            float score = Math.Min((float)Math.Exp(prediction.Score), 1.0f);
            float gold = 0.0f;
            if (Contains(labels, prediction.Label))
            {
                lm.PredictedGold++;
                _metrics.PredictedGold++;
                gold = 1.0f;
            }
            lm.ScoreVsTrue.Add((score, gold));
        }

        foreach (int label in labels)
        {
            LabelMetrics lm = GetLabel(label);
            lm.Gold++;
            if (_falseNegativeLabels && !ContainsSecond(predictions, label))
            {
                lm.ScoreVsTrue.Add((-1.0f, 1.0f));
            }
        }
    }

    /// <summary>Overall precision across all labels.</summary>
    public double Precision() => _metrics.Precision();

    /// <summary>Overall recall across all labels.</summary>
    public double Recall() => _metrics.Recall();

    /// <summary>Overall F1 score across all labels.</summary>
    public double F1Score()
    {
        double precision = Precision();
        double recall = Recall();
        if (precision + recall != 0)
        {
            return 2 * precision * recall / (precision + recall);
        }
        return double.NaN;
    }

    /// <summary>Precision for a single label id.</summary>
    public double Precision(int labelId) => GetLabel(labelId).Precision();

    /// <summary>Recall for a single label id.</summary>
    public double Recall(int labelId) => GetLabel(labelId).Recall();

    /// <summary>F1 score for a single label id.</summary>
    public double F1Score(int labelId) => GetLabel(labelId).F1Score();

    /// <summary>Precision, recall, and F1 for a single label id.</summary>
    public FastTextMetrics GetMetrics(int labelId)
    {
        LabelMetrics lm = GetLabel(labelId);
        return new FastTextMetrics(lm.Precision(), lm.Recall(), lm.F1Score());
    }

    private LabelMetrics GetLabel(int labelId)
    {
        if (!_labelMetrics.TryGetValue(labelId, out LabelMetrics? lm))
        {
            lm = new LabelMetrics();
            _labelMetrics[labelId] = lm;
        }
        return lm;
    }

    private static bool Contains(IReadOnlyList<int> labels, int value)
    {
        for (int i = 0; i < labels.Count; i++)
        {
            if (labels[i] == value)
            {
                return true;
            }
        }
        return false;
    }

    private static bool ContainsSecond(IReadOnlyList<Prediction> predictions, int value)
    {
        for (int i = 0; i < predictions.Count; i++)
        {
            if (predictions[i].Label == value)
            {
                return true;
            }
        }
        return false;
    }
}
