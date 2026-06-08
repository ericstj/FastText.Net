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

    // Training-only arguments (not persisted in the model file).
    public double Lr = 0.05;
    public int MinCountLabel = 0;
    public int Thread = 12;
    public int Seed = 0;

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

    public void Save(BinaryWriter writer)
    {
        writer.Write(Dim);
        writer.Write(Ws);
        writer.Write(Epoch);
        writer.Write(MinCount);
        writer.Write(Neg);
        writer.Write(WordNgrams);
        writer.Write((int)Loss);
        writer.Write((int)Model);
        writer.Write(Bucket);
        writer.Write(Minn);
        writer.Write(Maxn);
        writer.Write(LrUpdateRate);
        writer.Write(T);
    }

    public static string LossToString(LossName loss) => loss switch
    {
        LossName.Hs => "hs",
        LossName.Ns => "ns",
        LossName.Softmax => "softmax",
        LossName.Ova => "one-vs-all",
        _ => "Unknown loss!",
    };

    public static string ModelToString(ModelName model) => model switch
    {
        ModelName.Cbow => "cbow",
        ModelName.Sg => "sg",
        ModelName.Sup => "sup",
        _ => "Unknown model name!",
    };

    public void Dump(TextWriter o)
    {
        var c = System.Globalization.CultureInfo.InvariantCulture;
        o.WriteLine($"dim {Dim}");
        o.WriteLine($"ws {Ws}");
        o.WriteLine($"epoch {Epoch}");
        o.WriteLine($"minCount {MinCount}");
        o.WriteLine($"neg {Neg}");
        o.WriteLine($"wordNgrams {WordNgrams}");
        o.WriteLine($"loss {LossToString(Loss)}");
        o.WriteLine($"model {ModelToString(Model)}");
        o.WriteLine($"bucket {Bucket}");
        o.WriteLine($"minn {Minn}");
        o.WriteLine($"maxn {Maxn}");
        o.WriteLine($"lrUpdateRate {LrUpdateRate}");
        o.WriteLine(string.Create(c, $"t {T}"));
    }
}
