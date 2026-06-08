using FastTextNet;
using Xunit;
using FT = FastTextNet.FastText;

namespace FastText.Net.Tests;

// Validates quantization authoring (fastText's quantize). Product-quantization
// k-means is seeded but its results depend on the std-library shuffle/uniform
// implementations, so the .NET output is not byte-identical to the original C++
// model. Instead these tests assert the compression is lossy-but-faithful:
// the model reports as quantized, top-1 predictions still agree with the dense
// model, word vectors stay highly correlated, and a quantize -> save -> reload
// cycle is deterministic.
public sealed class QuantizationTests
{
    private static readonly string[] Inputs =
    {
        "hello good morning how are you",
        "the software code runs on a fast computer",
        "warm sunny weather today",
    };

    private static string SyntheticPath(string modelFile) =>
        Path.Combine(AppContext.BaseDirectory, "synthetic", modelFile);

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void QuantizePreservesTopPredictions(bool qnorm)
    {
        FT dense = FT.LoadModel(SyntheticPath("softmax.bin"));
        FT quant = FT.LoadModel(SyntheticPath("softmax.bin"));

        quant.Quantize(qnorm: qnorm);

        Assert.False(dense.IsQuantized);
        Assert.True(quant.IsQuantized);
        Assert.Equal(dense.Dimension, quant.Dimension);

        foreach (string text in Inputs)
        {
            string denseTop = dense.Predict(text, k: 1)[0].Label;
            string quantTop = quant.Predict(text, k: 1)[0].Label;
            Assert.Equal(denseTop, quantTop);
        }
    }

    [Fact]
    public void QuantizeKeepsWordVectorsCorrelated()
    {
        FT dense = FT.LoadModel(SyntheticPath("softmax.bin"));
        FT quant = FT.LoadModel(SyntheticPath("softmax.bin"));
        quant.Quantize();

        int compared = 0;
        foreach (string word in dense.GetWords())
        {
            float[] a = dense.GetWordVector(word);
            float[] b = quant.GetWordVector(word);
            double cos = Cosine(a, b);
            Assert.True(cos > 0.9, $"cosine for '{word}' was {cos:F3}");
            if (++compared == 20)
            {
                break;
            }
        }
        Assert.True(compared > 0);
    }

    [Fact]
    public void QuantizedModelRoundTripsDeterministically()
    {
        FT quant = FT.LoadModel(SyntheticPath("softmax.bin"));
        quant.Quantize();

        using var ms = new MemoryStream();
        quant.SaveModel(ms);
        ms.Position = 0;
        FT reloaded = FT.LoadModel(ms);

        Assert.True(reloaded.IsQuantized);
        foreach (string text in Inputs)
        {
            IReadOnlyList<FastTextPrediction> a = quant.Predict(text, k: 4);
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
    public void QuantizeWithCutoffPrunesVocabulary()
    {
        FT dense = FT.LoadModel(SyntheticPath("softmax.bin"));
        int fullVocab = dense.GetWords().Count;

        FT quant = FT.LoadModel(SyntheticPath("softmax.bin"));
        // Keep enough embeddings to satisfy the 256-row k-means minimum while
        // still pruning the full input matrix (words + 1000 ngram buckets).
        quant.Quantize(cutoff: 300);

        Assert.True(quant.IsQuantized);
        Assert.True(quant.GetWords().Count <= fullVocab);
        Assert.NotEmpty(quant.Predict(Inputs[0], k: 1));
    }

    [Fact]
    public void QuantizeRejectsAlreadyQuantizedModels()
    {
        FT quant = FT.LoadModel(SyntheticPath("softmax.bin"));
        quant.Quantize();
        Assert.Throws<InvalidOperationException>(() => quant.Quantize());
    }

    private static double Cosine(float[] a, float[] b)
    {
        double dot = 0, na = 0, nb = 0;
        for (int i = 0; i < a.Length; i++)
        {
            dot += a[i] * b[i];
            na += a[i] * a[i];
            nb += b[i] * b[i];
        }
        return dot / (Math.Sqrt(na) * Math.Sqrt(nb) + 1e-12);
    }
}
