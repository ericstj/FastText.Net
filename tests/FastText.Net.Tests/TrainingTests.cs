using FastTextNet;
using Xunit;
using FT = FastTextNet.FastText;

namespace FastText.Net.Tests;

// Validates the training pipeline (Phase 4). Trained models are not byte-identical
// to native fastText (std-library RNG/distributions differ), so training is verified
// by quality (the model learns its corpus), round-trip (save/load match), and
// determinism (a fixed seed reproduces the same model).
public sealed class TrainingTests
{
    private static string WriteCorpus(string contents)
    {
        string path = Path.Combine(Path.GetTempPath(), $"fasttext_{Guid.NewGuid():N}.txt");
        File.WriteAllText(path, contents);
        return path;
    }

    // Two linearly-separable classes: each label owns a disjoint vocabulary.
    private static string SupervisedCorpus()
    {
        var lines = new List<string>();
        for (int i = 0; i < 40; i++)
        {
            lines.Add("__label__animal cat dog pet fur paw tail");
            lines.Add("__label__vehicle car truck road wheel engine fuel");
        }
        return string.Join('\n', lines) + '\n';
    }

    private static FastTextTrainArgs SupervisedArgs() => new()
    {
        Model = FastTextArchitecture.Supervised,
        Loss = FastTextLossFunction.Softmax,
        Dim = 16,
        Lr = 0.5,
        Epoch = 20,
        MinCount = 1,
        Minn = 0,
        Maxn = 0,
        WordNgrams = 1,
        Bucket = 10000,
        Thread = 12,
        Seed = 1,
    };

    [Fact]
    public void SupervisedModelLearnsCorpus()
    {
        string corpus = WriteCorpus(SupervisedCorpus());
        try
        {
            FT model = FT.Train(corpus, SupervisedArgs());

            Assert.True(model.IsSupervised);
            Assert.Equal("__label__animal", model.Predict("cat dog pet")[0].Label);
            Assert.Equal("__label__vehicle", model.Predict("car truck road")[0].Label);
        }
        finally
        {
            File.Delete(corpus);
        }
    }

    [Fact]
    public void SupervisedModelRoundTrips()
    {
        string corpus = WriteCorpus(SupervisedCorpus());
        try
        {
            FT model = FT.Train(corpus, SupervisedArgs());

            using var ms = new MemoryStream();
            model.SaveModel(ms);
            ms.Position = 0;
            FT reloaded = FT.LoadModel(ms);

            foreach (string text in new[] { "cat fur paw", "truck wheel engine", "pet road" })
            {
                var original = model.Predict(text, k: 2);
                var copy = reloaded.Predict(text, k: 2);
                Assert.Equal(original.Count, copy.Count);
                for (int i = 0; i < original.Count; i++)
                {
                    Assert.Equal(original[i].Label, copy[i].Label);
                    Assert.Equal(original[i].Probability, copy[i].Probability, 5);
                }
            }
        }
        finally
        {
            File.Delete(corpus);
        }
    }

    [Fact]
    public void TrainingIsDeterministicForFixedSeed()
    {
        string corpus = WriteCorpus(SupervisedCorpus());
        try
        {
            FT a = FT.Train(corpus, SupervisedArgs());
            FT b = FT.Train(corpus, SupervisedArgs());

            float[] va = a.GetWordVector("cat");
            float[] vb = b.GetWordVector("cat");
            Assert.Equal(va.Length, vb.Length);
            for (int i = 0; i < va.Length; i++)
            {
                Assert.Equal(va[i], vb[i]);
            }
        }
        finally
        {
            File.Delete(corpus);
        }
    }

    [Fact]
    public void SkipgramProducesNonTrivialVectors()
    {
        var lines = new List<string>();
        for (int i = 0; i < 50; i++)
        {
            lines.Add("the quick brown fox jumps over the lazy dog near the river bank");
        }
        string corpus = WriteCorpus(string.Join('\n', lines) + '\n');
        try
        {
            var args = FastTextTrainArgs.ForSkipgram();
            args.Dim = 16;
            args.Epoch = 5;
            args.MinCount = 1;
            args.Bucket = 10000;
            args.Seed = 1;

            FT model = FT.Train(corpus, args);
            Assert.False(model.IsSupervised);

            float[] vec = model.GetWordVector("fox");
            double norm = 0;
            foreach (float f in vec)
            {
                norm += f * (double)f;
            }
            Assert.True(norm > 0, "word vector should be non-zero after training");
        }
        finally
        {
            File.Delete(corpus);
        }
    }
}
