using System.Text.Json;
using FastTextNet;
using Xunit;

namespace FastText.Net.Tests;

public sealed class OracleCase
{
    public string text { get; set; } = "";
    public List<string> labels { get; set; } = new();
    public List<double> probs { get; set; } = new();
}

public sealed class LanguageIdentificationTests
{
    private static string ModelPath()
    {
        string baseDir = AppContext.BaseDirectory;
        string local = Path.Combine(baseDir, "models", "lid.176.ftz");
        if (File.Exists(local))
        {
            return local;
        }
        string? env = Environment.GetEnvironmentVariable("FASTTEXT_LID_MODEL");
        if (!string.IsNullOrEmpty(env) && File.Exists(env))
        {
            return env;
        }
        throw new SkipException(
            "lid.176.ftz not found. Place it under tests/.../models or set FASTTEXT_LID_MODEL.");
    }

    private static List<OracleCase> LoadOracle()
    {
        string path = Path.Combine(AppContext.BaseDirectory, "oracle_lid.json");
        using FileStream fs = File.OpenRead(path);
        return JsonSerializer.Deserialize<List<OracleCase>>(fs)!;
    }

    [Fact]
    public void ModelLoads()
    {
        var model = FastTextModel.Load(ModelPath());
        Assert.True(model.IsQuantized);
        Assert.Equal(176, model.LabelCount);
        Assert.Equal(16, model.Dimension);
    }

    [Fact]
    public void MatchesReferenceTopLabel()
    {
        var model = FastTextModel.Load(ModelPath());
        foreach (OracleCase c in LoadOracle())
        {
            IReadOnlyList<FastTextPrediction> p = model.Predict(c.text, k: 5);
            Assert.NotEmpty(p);
            Assert.Equal(c.labels[0], p[0].Label);
        }
    }

    [Fact]
    public void MatchesReferenceLabelsAndProbabilities()
    {
        var model = FastTextModel.Load(ModelPath());
        foreach (OracleCase c in LoadOracle())
        {
            IReadOnlyList<FastTextPrediction> p = model.Predict(c.text, k: 5);
            int n = Math.Min(c.labels.Count, p.Count);
            for (int i = 0; i < n; i++)
            {
                Assert.Equal(c.labels[i], p[i].Label);
                Assert.True(
                    Math.Abs(c.probs[i] - p[i].Probability) < 1e-4,
                    $"[{c.text}] label {p[i].Label}: expected {c.probs[i]}, got {p[i].Probability}");
            }
        }
    }
}

[System.Diagnostics.CodeAnalysis.SuppressMessage("Design", "CA1032")]
public sealed class SkipException(string message) : Exception(message);
