using FastTextNet;
using Xunit;
using FT = FastTextNet.FastText;

namespace FastText.Net.Tests;

// Validates model serialization. fastText's saveModel is the exact inverse of
// loadModel, so a load -> save round-trip of a dense (unpruned) model must be
// byte-identical. The quantized lid model uses a pruneidx map whose on-disk
// order is not defined, so it is validated semantically (predictions match).
public sealed class SerializationTests
{
    private static string SyntheticPath(string modelFile) =>
        Path.Combine(AppContext.BaseDirectory, "synthetic", modelFile);

    public static IEnumerable<object[]> DenseModels()
    {
        yield return new object[] { "softmax.bin" };
        yield return new object[] { "ns.bin" };
        yield return new object[] { "ova.bin" };
        yield return new object[] { "hs.bin" };
    }

    [Theory]
    [MemberData(nameof(DenseModels))]
    public void RoundTripsByteIdentical(string modelFile)
    {
        string path = SyntheticPath(modelFile);
        byte[] original = File.ReadAllBytes(path);

        FT model = FT.LoadModel(path);
        using var ms = new MemoryStream();
        model.SaveModel(ms);

        Assert.Equal(original, ms.ToArray());
    }

    [Theory]
    [MemberData(nameof(DenseModels))]
    public void ReloadProducesIdenticalPredictions(string modelFile)
    {
        string path = SyntheticPath(modelFile);
        FT original = FT.LoadModel(path);

        using var ms = new MemoryStream();
        original.SaveModel(ms);
        ms.Position = 0;
        FT reloaded = FT.LoadModel(ms);

        string[] inputs =
        {
            "hello good morning how are you",
            "the software code runs on a fast computer",
            "warm sunny weather today",
        };
        foreach (string text in inputs)
        {
            IReadOnlyList<FastTextPrediction> a = original.Predict(text, k: 4);
            IReadOnlyList<FastTextPrediction> b = reloaded.Predict(text, k: 4);
            Assert.Equal(a.Count, b.Count);
            for (int i = 0; i < a.Count; i++)
            {
                Assert.Equal(a[i].Label, b[i].Label);
                Assert.Equal(a[i].Probability, b[i].Probability);
            }
        }
    }

    [Fact]
    public void QuantizedModelRoundTripsSemantically()
    {
        string baseDir = AppContext.BaseDirectory;
        string path = Path.Combine(baseDir, "models", "lid.176.ftz");
        if (!File.Exists(path))
        {
            throw new SkipException("lid.176.ftz not found.");
        }

        FT original = FT.LoadModel(path);
        using var ms = new MemoryStream();
        original.SaveModel(ms);
        ms.Position = 0;
        FT reloaded = FT.LoadModel(ms);

        Assert.True(reloaded.IsQuantized);
        string[] inputs =
        {
            "Hello, how are you doing today?",
            "Bonjour, comment allez-vous aujourd'hui?",
            "こんにちは、お元気ですか",
        };
        foreach (string text in inputs)
        {
            IReadOnlyList<FastTextPrediction> a = original.Predict(text, k: 5);
            IReadOnlyList<FastTextPrediction> b = reloaded.Predict(text, k: 5);
            Assert.Equal(a.Count, b.Count);
            for (int i = 0; i < a.Count; i++)
            {
                Assert.Equal(a[i].Label, b[i].Label);
                Assert.Equal(a[i].Probability, b[i].Probability);
            }
        }
    }
}
