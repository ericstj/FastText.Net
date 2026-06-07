using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Running;
using FastTextNet;

namespace FastTextNet.Benchmarks;

[MemoryDiagnoser]
public class LanguageIdBenchmarks
{
    private FastTextModel _model = null!;

    private const string ShortEnglish = "Hello, how are you doing today?";
    private const string LongEnglish =
        "fastText is a library for efficient learning of word representations and " +
        "sentence classification, supporting more than one hundred and fifty languages.";

    private static readonly string[] Corpus =
    {
        "Hello, how are you doing today?",
        "Bonjour, comment allez-vous aujourd'hui?",
        "Hola, ¿cómo estás hoy?",
        "Guten Tag, wie geht es Ihnen heute?",
        "Ciao, come stai oggi?",
        "Привет, как у тебя дела сегодня?",
        "こんにちは、お元気ですか",
        "你好，今天过得怎么样",
        "안녕하세요 오늘 기분이 어떠세요",
        "مرحبا كيف حالك اليوم",
    };

    [GlobalSetup]
    public void Setup() => _model = FastTextModel.Load(ModelLocator.Path());

    [Benchmark(Baseline = true)]
    public object PredictShort_K1() => _model.Predict(ShortEnglish, k: 1);

    [Benchmark]
    public object PredictShort_K5() => _model.Predict(ShortEnglish, k: 5);

    [Benchmark]
    public object PredictLong_K1() => _model.Predict(LongEnglish, k: 1);

    [Benchmark]
    public int PredictCorpus_K1()
    {
        int n = 0;
        foreach (string s in Corpus)
        {
            n += _model.Predict(s, k: 1).Count;
        }
        return n;
    }
}

public class LoadBenchmarks
{
    [Benchmark]
    public FastTextModel LoadModel() => FastTextModel.Load(ModelLocator.Path());
}

internal static class ModelLocator
{
    public static string Path()
    {
        string local = System.IO.Path.Combine(AppContext.BaseDirectory, "models", "lid.176.ftz");
        if (File.Exists(local))
        {
            return local;
        }
        string? env = Environment.GetEnvironmentVariable("FASTTEXT_LID_MODEL");
        if (!string.IsNullOrEmpty(env) && File.Exists(env))
        {
            return env;
        }
        throw new FileNotFoundException(
            "lid.176.ftz not found. Place it under bench/.../models or set FASTTEXT_LID_MODEL.");
    }
}

public static class Program
{
    public static void Main(string[] args) =>
        BenchmarkSwitcher.FromAssembly(typeof(Program).Assembly).Run(args);
}
