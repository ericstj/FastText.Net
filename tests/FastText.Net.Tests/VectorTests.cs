using System.Text.Json;
using FastTextNet;
using Xunit;
using FT = FastTextNet.FastText;

namespace FastText.Net.Tests;

// Validates the full-surface FastText facade (word/subword/sentence vectors,
// ids, and nearest neighbours) against reference fastText. Fixtures are the
// committed synthetic models; the oracle is produced by gen_vectors_oracle.py
// without retraining, so it tracks the exact .bin bytes under test.
public sealed class VectorTests
{
    private const float Tolerance = 1e-4f;

    public sealed class Neighbor
    {
        public string word { get; set; } = "";
        public float sim { get; set; }
    }

    public sealed class VectorOracle
    {
        public string model { get; set; } = "";
        public int dim { get; set; }
        public Dictionary<string, float[]> wordVectors { get; set; } = new();
        public Dictionary<string, int> wordIds { get; set; } = new();
        public Dictionary<string, int> subwordIds { get; set; } = new();
        public Dictionary<string, float[]> subwordVectors { get; set; } = new();
        public Dictionary<string, float[]> sentenceVectors { get; set; } = new();
        public Dictionary<string, List<Neighbor>> nearestNeighbors { get; set; } = new();
    }

    private static List<VectorOracle> LoadManifest()
    {
        string path = Path.Combine(AppContext.BaseDirectory, "vectors_oracle.json");
        using FileStream fs = File.OpenRead(path);
        return JsonSerializer.Deserialize<List<VectorOracle>>(fs)!;
    }

    public static IEnumerable<object[]> Models()
    {
        foreach (VectorOracle o in LoadManifest())
        {
            yield return new object[] { o.model };
        }
    }

    private static VectorOracle Get(string modelFile) =>
        LoadManifest().Single(o => o.model == modelFile);

    private static string ModelPath(string modelFile) =>
        Path.Combine(AppContext.BaseDirectory, "synthetic", modelFile);

    private static void AssertVectorsEqual(string context, float[] expected, float[] actual)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            Assert.True(
                Math.Abs(expected[i] - actual[i]) < Tolerance,
                $"{context}[{i}]: expected {expected[i]}, got {actual[i]}");
        }
    }

    [Theory]
    [MemberData(nameof(Models))]
    public void WordVectorsMatchReference(string modelFile)
    {
        VectorOracle oracle = Get(modelFile);
        FT model = FT.LoadModel(ModelPath(modelFile));

        Assert.Equal(oracle.dim, model.Dimension);
        foreach ((string word, float[] expected) in oracle.wordVectors)
        {
            AssertVectorsEqual($"[{modelFile}] word '{word}'", expected, model.GetWordVector(word));
        }
    }

    [Theory]
    [MemberData(nameof(Models))]
    public void SubwordVectorsMatchReference(string modelFile)
    {
        VectorOracle oracle = Get(modelFile);
        FT model = FT.LoadModel(ModelPath(modelFile));

        foreach ((string subword, float[] expected) in oracle.subwordVectors)
        {
            AssertVectorsEqual(
                $"[{modelFile}] subword '{subword}'", expected, model.GetSubwordVector(subword));
        }
    }

    [Theory]
    [MemberData(nameof(Models))]
    public void SentenceVectorsMatchReference(string modelFile)
    {
        VectorOracle oracle = Get(modelFile);
        FT model = FT.LoadModel(ModelPath(modelFile));

        foreach ((string text, float[] expected) in oracle.sentenceVectors)
        {
            AssertVectorsEqual(
                $"[{modelFile}] sentence '{text}'", expected, model.GetSentenceVector(text));
        }
    }

    [Theory]
    [MemberData(nameof(Models))]
    public void WordAndSubwordIdsMatchReference(string modelFile)
    {
        VectorOracle oracle = Get(modelFile);
        FT model = FT.LoadModel(ModelPath(modelFile));

        foreach ((string word, int expected) in oracle.wordIds)
        {
            Assert.Equal(expected, model.GetWordId(word));
        }
        foreach ((string subword, int expected) in oracle.subwordIds)
        {
            Assert.Equal(expected, model.GetSubwordId(subword));
        }
    }

    [Theory]
    [MemberData(nameof(Models))]
    public void NearestNeighborsMatchReference(string modelFile)
    {
        VectorOracle oracle = Get(modelFile);
        FT model = FT.LoadModel(ModelPath(modelFile));

        foreach ((string word, List<Neighbor> expected) in oracle.nearestNeighbors)
        {
            IReadOnlyList<FastTextNeighbor> actual = model.GetNearestNeighbors(word, k: expected.Count);
            Assert.Equal(expected.Count, actual.Count);

            // Compare as a similarity map: near-tied similarities can reorder
            // between implementations, but the word set and values must match.
            var actualByWord = actual.ToDictionary(n => n.Word, n => n.Similarity);
            foreach (Neighbor e in expected)
            {
                Assert.True(
                    actualByWord.TryGetValue(e.word, out float sim),
                    $"[{modelFile}] NN('{word}') missing '{e.word}'");
                Assert.True(
                    Math.Abs(e.sim - sim) < Tolerance,
                    $"[{modelFile}] NN('{word}') '{e.word}': expected {e.sim}, got {sim}");
            }
        }
    }
}
