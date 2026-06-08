// Ported from Facebook fastText (src/matrix.h, src/densematrix.*, src/quantmatrix.*).
// See THIRD-PARTY-NOTICES.md. Original: Copyright (c) Facebook, Inc. (MIT License).

using System.Numerics.Tensors;

namespace FastTextNet;

internal abstract class Matrix
{
    public long Rows { get; protected set; }
    public long Cols { get; protected set; }

    public abstract float DotRow(ReadOnlySpan<float> vec, int i);
    public abstract void AddRowToVector(Span<float> x, int i, float a);
    public abstract void AverageRowsToVector(Span<float> x, List<int> rows);
    public abstract void Load(BinaryReader reader);
    public abstract void Save(BinaryWriter writer);
}

internal sealed class DenseMatrix : Matrix
{
    private float[] _data = Array.Empty<float>();

    public DenseMatrix() { }

    public DenseMatrix(long rows, long cols)
    {
        Rows = rows;
        Cols = cols;
        _data = new float[checked(rows * cols)];
    }

    public void Zero() => Array.Clear(_data);

    public void AddVectorToRow(ReadOnlySpan<float> vec, int i, float a)
    {
        int n = (int)Cols;
        Span<float> row = _data.AsSpan(i * n, n);
        TensorPrimitives.MultiplyAdd(vec, a, row, row);
    }

    public override float DotRow(ReadOnlySpan<float> vec, int i)
    {
        int n = (int)Cols;
        return TensorPrimitives.Dot(_data.AsSpan(i * n, n), vec);
    }

    public override void AddRowToVector(Span<float> x, int i, float a)
    {
        int n = (int)Cols;
        ReadOnlySpan<float> row = _data.AsSpan(i * n, n);
        // x = row * a + x
        TensorPrimitives.MultiplyAdd(row, a, x, x);
    }

    public override void AverageRowsToVector(Span<float> x, List<int> rows)
    {
        int n = (int)Cols;
        if (rows.Count == 0)
        {
            x.Clear();
            return;
        }
        ReadOnlySpan<float> first = _data.AsSpan(rows[0] * n, n);
        first.CopyTo(x);
        for (int r = 1; r < rows.Count; r++)
        {
            ReadOnlySpan<float> row = _data.AsSpan(rows[r] * n, n);
            TensorPrimitives.Add(x, row, x);
        }
        TensorPrimitives.Multiply(x, 1.0f / rows.Count, x);
    }

    public override void Load(BinaryReader reader)
    {
        Rows = reader.ReadInt64();
        Cols = reader.ReadInt64();
        long count = Rows * Cols;
        _data = new float[count];
        byte[] bytes = reader.ReadBytes(checked((int)(count * sizeof(float))));
        Buffer.BlockCopy(bytes, 0, _data, 0, bytes.Length);
    }

    public override void Save(BinaryWriter writer)
    {
        writer.Write(Rows);
        writer.Write(Cols);
        byte[] bytes = new byte[checked(_data.Length * sizeof(float))];
        Buffer.BlockCopy(_data, 0, bytes, 0, bytes.Length);
        writer.Write(bytes);
    }
}

internal sealed class QuantMatrix : Matrix
{
    private ProductQuantizer _pq = new();
    private ProductQuantizer? _npq;
    private byte[] _codes = Array.Empty<byte>();
    private byte[] _normCodes = Array.Empty<byte>();
    private bool _qnorm;
    private int _codesize;

    public override float DotRow(ReadOnlySpan<float> vec, int i)
    {
        float norm = _qnorm ? _npq!.GetCentroid(0, _normCodes[i]) : 1f;
        return _pq.MulCode(vec, _codes, i, norm);
    }

    public override void AddRowToVector(Span<float> x, int i, float a)
    {
        float norm = _qnorm ? _npq!.GetCentroid(0, _normCodes[i]) : 1f;
        _pq.AddCode(x, _codes, i, a * norm);
    }

    public override void AverageRowsToVector(Span<float> x, List<int> rows)
    {
        x.Clear();
        if (rows.Count == 0)
        {
            return;
        }
        foreach (int r in rows)
        {
            AddRowToVector(x, r, 1f);
        }
        TensorPrimitives.Multiply(x, 1.0f / rows.Count, x);
    }

    public override void Load(BinaryReader reader)
    {
        _qnorm = reader.ReadBoolean();
        Rows = reader.ReadInt64();
        Cols = reader.ReadInt64();
        _codesize = reader.ReadInt32();
        _codes = reader.ReadBytes(_codesize);
        _pq = new ProductQuantizer();
        _pq.Load(reader);
        if (_qnorm)
        {
            _normCodes = reader.ReadBytes((int)Rows);
            _npq = new ProductQuantizer();
            _npq.Load(reader);
        }
    }

    public override void Save(BinaryWriter writer)
    {
        writer.Write(_qnorm);
        writer.Write(Rows);
        writer.Write(Cols);
        writer.Write(_codesize);
        writer.Write(_codes);
        _pq.Save(writer);
        if (_qnorm)
        {
            writer.Write(_normCodes);
            _npq!.Save(writer);
        }
    }
}
