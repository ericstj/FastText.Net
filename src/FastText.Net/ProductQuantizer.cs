// Ported from Facebook fastText (src/productquantizer.h, src/productquantizer.cc).
// See THIRD-PARTY-NOTICES.md. Original: Copyright (c) Facebook, Inc. (MIT License).

namespace FastTextNet;

/// <summary>
/// Product quantizer used by quantized (.ftz) models. Only the decode path required for
/// inference (centroid lookup, dot/add against codes) is ported; k-means training is not.
/// </summary>
internal sealed class ProductQuantizer
{
    private const int Nbits = 8;
    private const int Ksub = 1 << Nbits; // 256

    private int _dim;
    private int _nsubq;
    private int _dsub;
    private int _lastdsub;
    private float[] _centroids = Array.Empty<float>();

    private int CentroidOffset(int m, byte code) =>
        m == _nsubq - 1
            ? m * Ksub * _dsub + code * _lastdsub
            : (m * Ksub + code) * _dsub;

    public float GetCentroid(int m, byte code) => _centroids[CentroidOffset(m, code)];

    public float MulCode(ReadOnlySpan<float> x, byte[] codes, int t, float alpha)
    {
        float res = 0f;
        int d = _dsub;
        int codeBase = _nsubq * t;
        for (int m = 0; m < _nsubq; m++)
        {
            int c = CentroidOffset(m, codes[codeBase + m]);
            if (m == _nsubq - 1)
            {
                d = _lastdsub;
            }
            int xBase = m * _dsub;
            for (int n = 0; n < d; n++)
            {
                res += x[xBase + n] * _centroids[c + n];
            }
        }
        return res * alpha;
    }

    public void AddCode(Span<float> x, byte[] codes, int t, float alpha)
    {
        int d = _dsub;
        int codeBase = _nsubq * t;
        for (int m = 0; m < _nsubq; m++)
        {
            int c = CentroidOffset(m, codes[codeBase + m]);
            if (m == _nsubq - 1)
            {
                d = _lastdsub;
            }
            int xBase = m * _dsub;
            for (int n = 0; n < d; n++)
            {
                x[xBase + n] += alpha * _centroids[c + n];
            }
        }
    }

    public void Load(BinaryReader reader)
    {
        _dim = reader.ReadInt32();
        _nsubq = reader.ReadInt32();
        _dsub = reader.ReadInt32();
        _lastdsub = reader.ReadInt32();
        int count = _dim * Ksub;
        _centroids = new float[count];
        byte[] bytes = reader.ReadBytes(count * sizeof(float));
        Buffer.BlockCopy(bytes, 0, _centroids, 0, bytes.Length);
    }

    public void Save(BinaryWriter writer)
    {
        writer.Write(_dim);
        writer.Write(_nsubq);
        writer.Write(_dsub);
        writer.Write(_lastdsub);
        byte[] bytes = new byte[checked(_centroids.Length * sizeof(float))];
        Buffer.BlockCopy(_centroids, 0, bytes, 0, bytes.Length);
        writer.Write(bytes);
    }
}
