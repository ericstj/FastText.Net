// Ported from Facebook fastText (src/args.h, src/args.cc).
// See THIRD-PARTY-NOTICES.md. Original: Copyright (c) Facebook, Inc. (MIT License).

namespace FastTextNet;

internal enum ModelName { Cbow = 1, Sg = 2, Sup = 3 }

internal enum LossName { Hs = 1, Ns = 2, Softmax = 3, Ova = 4 }

/// <summary>
/// Subset of fastText training/model arguments that are persisted in a model file
/// and required for inference.
/// </summary>
internal sealed class Args
{
    public int Dim = 100;
    public int Ws = 5;
    public int Epoch = 5;
    public int MinCount = 5;
    public int Neg = 5;
    public int WordNgrams = 1;
    public LossName Loss = LossName.Ns;
    public ModelName Model = ModelName.Sg;
    public int Bucket = 2000000;
    public int Minn = 3;
    public int Maxn = 6;
    public int LrUpdateRate = 100;
    public double T = 1e-4;

    // Not persisted by Args.save but read separately from the model stream.
    public bool Qout;

    // The label prefix is not stored in the binary; fastText always uses this default
    // for pretrained supervised models.
    public string Label = "__label__";

    public void Load(BinaryReader reader)
    {
        Dim = reader.ReadInt32();
        Ws = reader.ReadInt32();
        Epoch = reader.ReadInt32();
        MinCount = reader.ReadInt32();
        Neg = reader.ReadInt32();
        WordNgrams = reader.ReadInt32();
        Loss = (LossName)reader.ReadInt32();
        Model = (ModelName)reader.ReadInt32();
        Bucket = reader.ReadInt32();
        Minn = reader.ReadInt32();
        Maxn = reader.ReadInt32();
        LrUpdateRate = reader.ReadInt32();
        T = reader.ReadDouble();
    }
}
