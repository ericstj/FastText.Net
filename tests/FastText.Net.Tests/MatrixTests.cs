using FastTextNet;
using Xunit;

namespace FastText.Net.Tests;

public sealed class MatrixTests
{
    [Theory]
    [InlineData(1)]
    [InlineData(4)]
    [InlineData(12)]
    public void UniformInitializesEntireMatrix(int thread)
    {
        var matrix = new DenseMatrix(7, 9);
        Array.Fill(matrix.RawData, float.NaN);

        matrix.Uniform(0.5f, thread, seed: 1);

        foreach (float value in matrix.RawData)
        {
            Assert.False(float.IsNaN(value), "Every element must be initialized.");
            Assert.InRange(value, -0.5f, 0.5f);
        }
    }
}
