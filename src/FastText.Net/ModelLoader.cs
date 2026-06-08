// Ported from Facebook fastText (src/fasttext.cc loadModel/signModel/checkModel).
// See THIRD-PARTY-NOTICES.md. Original: Copyright (c) Facebook, Inc. (MIT License).

using System.Text;

namespace FastTextNet;

internal sealed record LoadedModel(
    Args Args,
    Dictionary Dict,
    Matrix Input,
    Matrix Output,
    Model Model,
    bool Quantized);

internal static class ModelLoader
{
    private const int MagicInt32 = 793712314;
    private const int SupportedVersion = 12;

    public static LoadedModel Load(Stream stream)
    {        using var reader = new BinaryReader(stream, Encoding.UTF8, leaveOpen: true);

        int magic = reader.ReadInt32();
        if (magic != MagicInt32)
        {
            throw new InvalidDataException("Not a fastText model file (bad magic number).");
        }
        int version = reader.ReadInt32();
        if (version > SupportedVersion)
        {
            throw new InvalidDataException(
                $"Unsupported fastText model version {version} (supported up to {SupportedVersion}).");
        }

        var args = new Args();
        args.Load(reader);
        if (version == 11 && args.Model == ModelName.Sup)
        {
            args.Maxn = 0;
        }

        var dict = new Dictionary(args);
        dict.Load(reader);

        bool quantInput = reader.ReadBoolean();
        Matrix input = quantInput ? new QuantMatrix() : new DenseMatrix();
        input.Load(reader);

        if (!quantInput && dict.IsPruned)
        {
            throw new InvalidDataException(
                "Invalid model file: pruned dictionary without quantized input. " +
                "Please download an updated model from fasttext.cc.");
        }

        args.Qout = reader.ReadBoolean();
        Matrix output = quantInput && args.Qout ? new QuantMatrix() : new DenseMatrix();
        output.Load(reader);

        Loss loss = CreateLoss(args, dict, output);
        var model = new Model(input, output, loss);
        return new LoadedModel(args, dict, input, output, model, quantInput);
    }

    public static void Save(Stream stream, Args args, Dictionary dict, Matrix input, Matrix output, bool quantized)
    {
        using var writer = new BinaryWriter(stream, Encoding.UTF8, leaveOpen: true);
        writer.Write(MagicInt32);
        writer.Write(SupportedVersion);
        args.Save(writer);
        dict.Save(writer);
        writer.Write(quantized);
        input.Save(writer);
        writer.Write(args.Qout);
        output.Save(writer);
    }

    public static Model BuildModel(Args args, Dictionary dict, Matrix input, Matrix output) =>
        new(input, output, CreateLoss(args, dict, output));

    private static Loss CreateLoss(Args args, Dictionary dict, Matrix output)
    {
        IReadOnlyList<long> Counts() => args.Model == ModelName.Sup
            ? dict.GetCounts(EntryType.Label)
            : dict.GetCounts(EntryType.Word);

        return args.Loss switch
        {
            LossName.Hs => new HierarchicalSoftmaxLoss(output, Counts()),
            LossName.Ns => new SigmoidLoss(output),
            LossName.Softmax => new SoftmaxLoss(output),
            LossName.Ova => new SigmoidLoss(output),
            _ => throw new InvalidDataException("Unknown loss function in model."),
        };
    }
}
