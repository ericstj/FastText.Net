// Ported from Facebook fastText (src/vector.h, src/vector.cc).
// See THIRD-PARTY-NOTICES.md. Original: Copyright (c) Facebook, Inc. (MIT License).

using System.Numerics.Tensors;

namespace FastTextNet;

internal sealed class Vector
{
    private readonly float[] _data;

    public Vector(int size) => _data = new float[size];

    public int Size => _data.Length;

    public Span<float> Data => _data;

    public float this[int i]
    {
        get => _data[i];
        set => _data[i] = value;
    }

    public void Zero() => Array.Clear(_data);

    public void Mul(float a) => TensorPrimitives.Multiply(_data, a, _data);

    public float Norm() => TensorPrimitives.Norm(_data);

    public void AddVector(Vector source) => TensorPrimitives.Add(_data, source._data, _data);

    public void AddVector(Vector source, float s) =>
        TensorPrimitives.MultiplyAdd(source._data, s, _data, _data);

    public void AddRow(Matrix a, int i) => a.AddRowToVector(_data, i, 1f);

    public void AddRow(Matrix a, int i, float scale) => a.AddRowToVector(_data, i, scale);

    public void Mul(Matrix a, Vector vec)
    {
        for (int i = 0; i < Size; i++)
        {
            _data[i] = a.DotRow(vec._data, i);
        }
    }

    public int Argmax()
    {
        float max = _data[0];
        int argmax = 0;
        for (int i = 1; i < Size; i++)
        {
            if (_data[i] > max)
            {
                max = _data[i];
                argmax = i;
            }
        }
        return argmax;
    }
}
