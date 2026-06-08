using FastTextNet;
using Xunit;
using FT = FastTextNet.FastText;

namespace FastText.Net.Tests;

// Validates hyper-parameter autotuning (Phase 5). Autotune is a stochastic, time-bounded
// random search; tests use a fixed seed plus a MaxTrials cap for bounded, reproducible runs.
public sealed class AutotuneTests : IDisposable
{
    private readonly string _train;
    private readonly string _valid;

    public AutotuneTests()
    {
        _train = WriteCorpus();
        _valid = WriteCorpus();
    }

    public void Dispose()
    {
        File.Delete(_train);
        File.Delete(_valid);
    }

    private static string WriteCorpus()
    {
        var lines = new List<string>();
        for (int i = 0; i < 40; i++)
        {
            lines.Add("__label__animal cat dog pet fur paw tail");
            lines.Add("__label__vehicle car truck road wheel engine fuel");
        }
        string path = Path.Combine(Path.GetTempPath(), $"fasttext_at_{Guid.NewGuid():N}.txt");
        File.WriteAllText(path, string.Join('\n', lines) + '\n');
        return path;
    }

    private FastTextTrainArgs BaseArgs() => new()
    {
        Model = FastTextArchitecture.Supervised,
        Loss = FastTextLossFunction.Softmax,
        Dim = 16,
        Epoch = 5,
        MinCount = 1,
        Minn = 0,
        Maxn = 0,
        Bucket = 10000,
        Seed = 1,
    };

    [Fact]
    public void AutotunedModelClassifiesValidation()
    {
        var at = new FastTextAutotuneArgs
        {
            ValidationFile = _valid,
            Duration = 30,
            MaxTrials = 12,
            Seed = 1,
        };

        FT model = FT.TrainSupervisedAutotune(_train, at, BaseArgs());

        Assert.True(model.IsSupervised);
        Assert.Equal("__label__animal", model.Predict("cat dog pet")[0].Label);
        Assert.Equal("__label__vehicle", model.Predict("car truck road")[0].Label);
    }

    [Fact]
    public void AutotuneIsDeterministicForFixedSeed()
    {
        FastTextAutotuneArgs MakeArgs() => new()
        {
            ValidationFile = _valid,
            Duration = 30,
            MaxTrials = 10,
            Seed = 7,
        };

        FT a = FT.TrainSupervisedAutotune(_train, MakeArgs(), BaseArgs());
        FT b = FT.TrainSupervisedAutotune(_train, MakeArgs(), BaseArgs());

        foreach (string text in new[] { "cat fur", "truck wheel", "dog paw road" })
        {
            var pa = a.Predict(text);
            var pb = b.Predict(text);
            Assert.Equal(pa[0].Label, pb[0].Label);
            Assert.Equal(pa[0].Probability, pb[0].Probability, 5);
        }
    }

    [Fact]
    public void PinnedArgumentIsNotOptimized()
    {
        FastTextTrainArgs baseArgs = BaseArgs();
        baseArgs.Dim = 24;
        baseArgs.PinnedArgs.Add("dim");

        var at = new FastTextAutotuneArgs
        {
            ValidationFile = _valid,
            Duration = 30,
            MaxTrials = 8,
            Seed = 3,
        };

        FT model = FT.TrainSupervisedAutotune(_train, at, baseArgs);
        Assert.Equal(24, model.Dimension);
    }

    [Fact]
    public void ModelSizeConstraintProducesQuantizedModel()
    {
        // Quantization (product-quantizer k-means) needs >= 256 input rows, so this corpus has
        // a large vocabulary. Bucket is pinned to bound per-trial memory.
        string train = WriteLargeVocabCorpus();
        string valid = WriteLargeVocabCorpus();
        try
        {
            FastTextTrainArgs baseArgs = BaseArgs();
            baseArgs.Dim = 16;
            baseArgs.Bucket = 50000;
            baseArgs.PinnedArgs.Add("dim");
            baseArgs.PinnedArgs.Add("bucket");

            var at = new FastTextAutotuneArgs
            {
                ValidationFile = valid,
                Duration = 30,
                MaxTrials = 5,
                Seed = 2,
                ModelSize = 2_000_000,
            };

            FT model = FT.TrainSupervisedAutotune(train, at, baseArgs);
            Assert.True(model.IsQuantized);
        }
        finally
        {
            File.Delete(train);
            File.Delete(valid);
        }
    }

    private static string WriteLargeVocabCorpus()
    {
        var animal = new List<string>();
        var vehicle = new List<string>();
        for (int i = 0; i < 140; i++)
        {
            animal.Add($"an{i}");
            vehicle.Add($"ve{i}");
        }

        var lines = new List<string>();
        for (int line = 0; line < 140; line++)
        {
            var a = new List<string> { "__label__animal" };
            var v = new List<string> { "__label__vehicle" };
            for (int j = 0; j < 6; j++)
            {
                a.Add(animal[(line + j) % animal.Count]);
                v.Add(vehicle[(line + j) % vehicle.Count]);
            }
            lines.Add(string.Join(' ', a));
            lines.Add(string.Join(' ', v));
        }
        string path = Path.Combine(Path.GetTempPath(), $"fasttext_atq_{Guid.NewGuid():N}.txt");
        File.WriteAllText(path, string.Join('\n', lines) + '\n');
        return path;
    }
}
