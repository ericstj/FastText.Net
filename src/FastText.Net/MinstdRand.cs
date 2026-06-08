// Managed equivalent of C++ std::minstd_rand (the LCG fastText seeds its training
// RNG with). The sequence matches std::minstd_rand, but std::uniform_*_distribution
// implementations differ across standard libraries, so trained models are not
// byte-identical to the native ones (verified by quality/round-trip instead).

namespace FastTextNet;

internal struct MinstdRand
{
    private const uint Modulus = 2147483647; // 2^31 - 1
    private const uint Multiplier = 16807;

    private uint _state;

    public MinstdRand(int seed)
    {
        long s = seed % Modulus;
        if (s < 0)
        {
            s += Modulus;
        }
        // minstd cannot be seeded with 0; std clamps it to 1.
        _state = s == 0 ? 1u : (uint)s;
    }

    public uint Next()
    {
        _state = (uint)((_state * (ulong)Multiplier) % Modulus);
        return _state;
    }

    /// <summary>Uniform real in [0, 1), approximating std::uniform_real_distribution(0, 1).</summary>
    public float NextFloat() => (Next() - 1) / (float)(Modulus - 1);

    /// <summary>Uniform integer in [min, max], approximating std::uniform_int_distribution.</summary>
    public int NextInt(int min, int max) => min + (int)(Next() % (uint)(max - min + 1));
}
