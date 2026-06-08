// Ported from Facebook fastText (src/productquantizer.h, src/productquantizer.cc).
// See THIRD-PARTY-NOTICES.md. Original: Copyright (c) Facebook, Inc. (MIT License).

namespace FastTextNet;

/// <summary>
/// Product quantizer used by quantized (.ftz) models. Supports both the decode path used
/// for inference (centroid lookup, dot/add against codes) and k-means training used when
/// authoring a quantized model.
/// </summary>
internal sealed class ProductQuantizer
{
    private const int Nbits = 8;
    private const int Ksub = 1 << Nbits; // 256
    private const int MaxPointsPerCluster = 256;
    private const int MaxPoints = MaxPointsPerCluster * Ksub;
    private const int Seed = 1234;
    private const int Niter = 25;
    private const float Eps = 1e-7f;

    private int _dim;
    private int _nsubq;
    private int _dsub;
    private int _lastdsub;
    private float[] _centroids = Array.Empty<float>();

    private uint _rng = Seed;

    public ProductQuantizer() { }

    public ProductQuantizer(int dim, int dsub)
    {
        _dim = dim;
        _nsubq = dim / dsub;
        _dsub = dsub;
        _centroids = new float[dim * Ksub];
        _lastdsub = dim % dsub;
        if (_lastdsub == 0)
        {
            _lastdsub = dsub;
        }
        else
        {
            _nsubq++;
        }
    }

    private int NextRand()
    {
        _rng = (uint)((_rng * 16807UL) % 2147483647UL);
        return (int)_rng;
    }

    private double NextUniform() => NextRand() / 2147483647.0;

    private void Shuffle(int[] perm)
    {
        for (int i = perm.Length - 1; i > 0; i--)
        {
            int j = NextRand() % (i + 1);
            (perm[i], perm[j]) = (perm[j], perm[i]);
        }
    }

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

    private static float DistL2(float[] x, int xOff, float[] c, int cOff, int d)
    {
        float dist = 0f;
        for (int i = 0; i < d; i++)
        {
            float t = x[xOff + i] - c[cOff + i];
            dist += t * t;
        }
        return dist;
    }

    private byte AssignCentroid(float[] x, int xOff, int cBase, int d)
    {
        byte code = 0;
        float dis = DistL2(x, xOff, _centroids, cBase, d);
        for (int j = 1; j < Ksub; j++)
        {
            float disij = DistL2(x, xOff, _centroids, cBase + j * d, d);
            if (disij < dis)
            {
                code = (byte)j;
                dis = disij;
            }
        }
        return code;
    }

    private void Estep(float[] x, int cBase, byte[] codes, int d, int n)
    {
        for (int i = 0; i < n; i++)
        {
            codes[i] = AssignCentroid(x, i * d, cBase, d);
        }
    }

    private void MStep(float[] x, int cBase, byte[] codes, int d, int n)
    {
        int[] nelts = new int[Ksub];
        Array.Clear(_centroids, cBase, d * Ksub);
        for (int i = 0; i < n; i++)
        {
            int k = codes[i];
            int cOff = cBase + k * d;
            int xOff = i * d;
            for (int j = 0; j < d; j++)
            {
                _centroids[cOff + j] += x[xOff + j];
            }
            nelts[k]++;
        }

        for (int k = 0; k < Ksub; k++)
        {
            float z = nelts[k];
            if (z != 0)
            {
                int cOff = cBase + k * d;
                for (int j = 0; j < d; j++)
                {
                    _centroids[cOff + j] /= z;
                }
            }
        }

        for (int k = 0; k < Ksub; k++)
        {
            if (nelts[k] == 0)
            {
                int m = 0;
                while (NextUniform() * (n - Ksub) >= nelts[m] - 1)
                {
                    m = (m + 1) % Ksub;
                }
                Array.Copy(_centroids, cBase + m * d, _centroids, cBase + k * d, d);
                for (int j = 0; j < d; j++)
                {
                    int sign = (j % 2) * 2 - 1;
                    _centroids[cBase + k * d + j] += sign * Eps;
                    _centroids[cBase + m * d + j] -= sign * Eps;
                }
                nelts[k] = nelts[m] / 2;
                nelts[m] -= nelts[k];
            }
        }
    }

    private void Kmeans(float[] x, int cBase, int n, int d)
    {
        int[] perm = new int[n];
        for (int i = 0; i < n; i++)
        {
            perm[i] = i;
        }
        Shuffle(perm);
        for (int i = 0; i < Ksub; i++)
        {
            Array.Copy(x, perm[i] * d, _centroids, cBase + i * d, d);
        }
        byte[] codes = new byte[n];
        for (int i = 0; i < Niter; i++)
        {
            Estep(x, cBase, codes, d, n);
            MStep(x, cBase, codes, d, n);
        }
    }

    public void Train(int n, float[] x)
    {
        if (n < Ksub)
        {
            throw new ArgumentException(
                $"Matrix too small for quantization, must have at least {Ksub} rows", nameof(n));
        }
        int[] perm = new int[n];
        for (int i = 0; i < n; i++)
        {
            perm[i] = i;
        }
        int d = _dsub;
        int np = Math.Min(n, MaxPoints);
        float[] xslice = new float[np * _dsub];
        for (int m = 0; m < _nsubq; m++)
        {
            if (m == _nsubq - 1)
            {
                d = _lastdsub;
            }
            if (np != n)
            {
                Shuffle(perm);
            }
            for (int j = 0; j < np; j++)
            {
                Array.Copy(x, perm[j] * _dim + m * _dsub, xslice, j * d, d);
            }
            Kmeans(xslice, CentroidOffset(m, 0), np, d);
        }
    }

    public void ComputeCode(float[] x, int xOff, byte[] codes, int codeOff)
    {
        int d = _dsub;
        for (int m = 0; m < _nsubq; m++)
        {
            if (m == _nsubq - 1)
            {
                d = _lastdsub;
            }
            codes[codeOff + m] = AssignCentroid(x, xOff + m * _dsub, CentroidOffset(m, 0), d);
        }
    }

    public void ComputeCodes(float[] x, byte[] codes, int n)
    {
        for (int i = 0; i < n; i++)
        {
            ComputeCode(x, i * _dim, codes, i * _nsubq);
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
