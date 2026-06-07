using System.Text.Json;
using FastTextNet;
using Xunit;

namespace FastText.Net.Tests;

// Validates the prediction paths that lid.176.ftz (quantized + hierarchical
// softmax) does not exercise: dense input matrices combined with the softmax,
// negative-sampling, one-vs-all, and (dense) hierarchical-softmax losses, plus
// wordNgrams>1 and character subwords. Fixtures are tiny models trained by
// gen_synthetic.py, each paired with its own reference oracle.
public sealed class SyntheticModelTests
{
    public sealed class SyntheticOracle
    {
        public string model { get; set; } = "";
        public string loss { get; set; } = "";
        public int dim { get; set; }
        public int labelCount { get; set; }
        public List<OracleCase> cases { get; set; } = new();
    }

    private static List<SyntheticOracle> LoadManifest()
    {
        string path = Path.Combine(AppContext.BaseDirectory, "synthetic_oracle.json");
        using FileStream fs = File.OpenRead(path);
        return JsonSerializer.Deserialize<List<SyntheticOracle>>(fs)!;
    }

    public static IEnumerable<object[]> Models()
    {
        foreach (SyntheticOracle o in LoadManifest())
        {
            yield return new object[] { o.model };
        }
    }

    private static SyntheticOracle Get(string modelFile) =>
        LoadManifest().Single(o => o.model == modelFile);

    private static string ModelPath(string modelFile) =>
        Path.Combine(AppContext.BaseDirectory, "synthetic", modelFile);

    [Theory]
    [MemberData(nameof(Models))]
    public void MatchesReferenceLabelsAndProbabilities(string modelFile)
    {
        SyntheticOracle oracle = Get(modelFile);
        var model = FastTextModel.Load(ModelPath(modelFile));

        Assert.False(model.IsQuantized);
        Assert.Equal(oracle.dim, model.Dimension);
        Assert.Equal(oracle.labelCount, model.LabelCount);

        foreach (OracleCase c in oracle.cases)
        {
            IReadOnlyList<FastTextPrediction> p = model.Predict(c.text, k: oracle.labelCount);
            Assert.Equal(c.labels.Count, p.Count);

            // Compare probabilities per label rather than by rank: ns/ova produce
            // independent sigmoids that can tie exactly, and tie-break order is
            // implementation-defined. Validate the computed value for each label.
            var expected = new Dictionary<string, double>();
            for (int i = 0; i < c.labels.Count; i++)
            {
                expected[c.labels[i]] = c.probs[i];
            }

            foreach (FastTextPrediction pred in p)
            {
                Assert.True(
                    expected.TryGetValue(pred.Label, out double ep),
                    $"[{modelFile}:{oracle.loss}] '{c.text}' unexpected label {pred.Label}");
                Assert.True(
                    Math.Abs(ep - pred.Probability) < 1e-4,
                    $"[{modelFile}:{oracle.loss}] '{c.text}' label {pred.Label}: expected {ep}, got {pred.Probability}");
            }

            double maxProb = c.probs.Max();
            Assert.True(
                Math.Abs(p[0].Probability - maxProb) < 1e-4,
                $"[{modelFile}:{oracle.loss}] '{c.text}' top prob: expected {maxProb}, got {p[0].Probability}");
        }
    }
}
