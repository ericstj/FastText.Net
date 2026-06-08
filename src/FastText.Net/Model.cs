// Ported from Facebook fastText (src/model.h, src/model.cc, src/loss.h, src/loss.cc).
// See THIRD-PARTY-NOTICES.md. Original: Copyright (c) Facebook, Inc. (MIT License).

using System.Numerics.Tensors;

namespace FastTextNet;

internal sealed class ModelState
{
    public readonly float[] Hidden;
    public readonly float[] Output;
    public readonly float[] Grad;

    public MinstdRand Rng;

    private double _lossValue;
    private long _nexamples;

    public ModelState(int hiddenSize, int outputSize, int seed)
    {
        Hidden = new float[hiddenSize];
        Output = new float[outputSize];
        Grad = new float[hiddenSize];
        Rng = new MinstdRand(seed);
    }

    public float GetLoss() => _nexamples == 0 ? 0f : (float)(_lossValue / _nexamples);

    public void IncrementNExamples(float loss)
    {
        _lossValue += loss;
        _nexamples++;
    }
}

internal sealed class Model
{
    public const int AllLabelsAsTarget = -1;

    private readonly Matrix _wi;
    private readonly Matrix _wo;
    private readonly Loss _loss;
    private readonly bool _normalizeGradient;

    public Model(Matrix wi, Matrix wo, Loss loss, bool normalizeGradient = false)
    {
        _wi = wi;
        _wo = wo;
        _loss = loss;
        _normalizeGradient = normalizeGradient;
    }

    public int OutputSize => (int)_wo.Rows;

    private void ComputeHidden(List<int> input, ModelState state) =>
        _wi.AverageRowsToVector(state.Hidden, input);

    public void Predict(List<int> input, int k, float threshold, List<Prediction> heap, ModelState state)
    {
        if (k <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(k), "k needs to be 1 or higher!");
        }
        ComputeHidden(input, state);
        _loss.Predict(k, threshold, heap, state);
    }

    public void Update(List<int> input, List<int> targets, int targetIndex, float lr, ModelState state)
    {
        if (input.Count == 0)
        {
            return;
        }
        ComputeHidden(input, state);

        Array.Clear(state.Grad);
        float lossValue = _loss.Forward(targets, targetIndex, state, lr, backprop: true);
        state.IncrementNExamples(lossValue);

        if (_normalizeGradient)
        {
            TensorPrimitives.Multiply(state.Grad, 1.0f / input.Count, state.Grad);
        }
        foreach (int i in input)
        {
            _wi.AddVectorToRow(state.Grad, i, 1.0f);
        }
    }
}

internal readonly record struct Prediction(float Score, int Label);

internal abstract class Loss
{
    private const int SigmoidTableSize = 512;
    private const int MaxSigmoid = 8;
    private const int LogTableSize = 512;

    private static readonly float[] SigmoidTable = BuildSigmoidTable();
    private static readonly float[] LogTable = BuildLogTable();

    protected readonly Matrix Wo;

    protected Loss(Matrix wo) => Wo = wo;

    private static float[] BuildSigmoidTable()
    {
        var t = new float[SigmoidTableSize + 1];
        for (int i = 0; i < SigmoidTableSize + 1; i++)
        {
            float x = (float)(i * 2 * MaxSigmoid) / SigmoidTableSize - MaxSigmoid;
            t[i] = (float)(1.0 / (1.0 + Math.Exp(-x)));
        }
        return t;
    }

    private static float[] BuildLogTable()
    {
        var t = new float[LogTableSize + 1];
        for (int i = 0; i < LogTableSize + 1; i++)
        {
            float x = (float)((i + 1e-5) / LogTableSize);
            t[i] = (float)Math.Log(x);
        }
        return t;
    }

    protected static float Sigmoid(float x)
    {
        if (x < -MaxSigmoid)
        {
            return 0f;
        }
        if (x > MaxSigmoid)
        {
            return 1f;
        }
        int i = (int)((x + MaxSigmoid) * SigmoidTableSize / MaxSigmoid / 2);
        return SigmoidTable[i];
    }

    protected static float Log(float x)
    {
        if (x > 1.0f)
        {
            return 0f;
        }
        int i = (int)(x * LogTableSize);
        return LogTable[i];
    }

    protected static float StdLog(float x) => (float)Math.Log((double)x + 1e-5);

    public abstract void ComputeOutput(ModelState state);

    public abstract float Forward(List<int> targets, int targetIndex, ModelState state, float lr, bool backprop);

    public virtual void Predict(int k, float threshold, List<Prediction> heap, ModelState state)
    {
        ComputeOutput(state);
        FindKBest(k, threshold, heap, state.Output);
    }

    protected static void FindKBest(int k, float threshold, List<Prediction> heap, float[] output)
    {
        for (int i = 0; i < output.Length; i++)
        {
            if (output[i] < threshold)
            {
                continue;
            }
            heap.Add(new Prediction(StdLog(output[i]), i));
        }
        // Descending by score; ties resolved by ascending label index (deterministic).
        heap.Sort(static (a, b) =>
        {
            int c = b.Score.CompareTo(a.Score);
            return c != 0 ? c : a.Label.CompareTo(b.Label);
        });
        if (heap.Count > k)
        {
            heap.RemoveRange(k, heap.Count - k);
        }
    }
}

internal abstract class BinaryLogisticLoss : Loss
{
    protected BinaryLogisticLoss(Matrix wo) : base(wo) { }

    protected float BinaryLogistic(int target, ModelState state, bool labelIsPositive, float lr, bool backprop)
    {
        float score = Sigmoid(Wo.DotRow(state.Hidden, target));
        if (backprop)
        {
            float alpha = lr * ((labelIsPositive ? 1f : 0f) - score);
            Wo.AddRowToVector(state.Grad, target, alpha);
            Wo.AddVectorToRow(state.Hidden, target, alpha);
        }
        return labelIsPositive ? -Log(score) : -Log(1.0f - score);
    }

    public override void ComputeOutput(ModelState state)
    {
        float[] output = state.Output;
        ReadOnlySpan<float> hidden = state.Hidden;
        for (int i = 0; i < output.Length; i++)
        {
            output[i] = Sigmoid(Wo.DotRow(hidden, i));
        }
    }
}

internal sealed class NegativeSamplingLoss : BinaryLogisticLoss
{
    private const int NegativeTableSize = 10000000;

    private readonly int _neg;
    private readonly IReadOnlyList<long> _targetCounts;
    private int[]? _negatives;

    public NegativeSamplingLoss(Matrix wo, int neg, IReadOnlyList<long> targetCounts) : base(wo)
    {
        _neg = neg;
        _targetCounts = targetCounts;
    }

    private int[] Negatives()
    {
        if (_negatives is not null)
        {
            return _negatives;
        }
        var negatives = new List<int>();
        double z = 0.0;
        for (int i = 0; i < _targetCounts.Count; i++)
        {
            z += Math.Pow(_targetCounts[i], 0.5);
        }
        for (int i = 0; i < _targetCounts.Count; i++)
        {
            double c = Math.Pow(_targetCounts[i], 0.5);
            for (int j = 0; j < c * NegativeTableSize / z; j++)
            {
                negatives.Add(i);
            }
        }
        _negatives = negatives.ToArray();
        return _negatives;
    }

    public override float Forward(List<int> targets, int targetIndex, ModelState state, float lr, bool backprop)
    {
        int target = targets[targetIndex];
        float loss = BinaryLogistic(target, state, true, lr, backprop);
        for (int n = 0; n < _neg; n++)
        {
            int negativeTarget = GetNegative(target, ref state.Rng);
            loss += BinaryLogistic(negativeTarget, state, false, lr, backprop);
        }
        return loss;
    }

    private int GetNegative(int target, ref MinstdRand rng)
    {
        int[] negatives = Negatives();
        int negative;
        do
        {
            negative = negatives[(int)(rng.Next() % (uint)negatives.Length)];
        }
        while (target == negative);
        return negative;
    }
}

internal sealed class OneVsAllLoss : BinaryLogisticLoss
{
    public OneVsAllLoss(Matrix wo) : base(wo) { }

    public override float Forward(List<int> targets, int targetIndex, ModelState state, float lr, bool backprop)
    {
        float loss = 0f;
        int osz = state.Output.Length;
        var targetSet = new HashSet<int>(targets);
        for (int i = 0; i < osz; i++)
        {
            bool isMatch = targetSet.Contains(i);
            loss += BinaryLogistic(i, state, isMatch, lr, backprop);
        }
        return loss;
    }
}

internal sealed class HierarchicalSoftmaxLoss : BinaryLogisticLoss
{
    private struct Node
    {
        public int Parent;
        public int Left;
        public int Right;
        public long Count;
        public bool Binary;
    }

    private readonly Node[] _tree;
    private readonly List<int>[] _paths;
    private readonly List<bool>[] _codes;
    private readonly int _osz;

    public HierarchicalSoftmaxLoss(Matrix wo, IReadOnlyList<long> counts) : base(wo)
    {
        _osz = counts.Count;
        _tree = new Node[2 * _osz - 1];
        _paths = new List<int>[_osz];
        _codes = new List<bool>[_osz];
        BuildTree(counts);
    }

    public override void ComputeOutput(ModelState state) =>
        throw new NotSupportedException("Hierarchical softmax uses tree traversal for prediction.");

    private void BuildTree(IReadOnlyList<long> counts)
    {
        for (int i = 0; i < 2 * _osz - 1; i++)
        {
            _tree[i].Parent = -1;
            _tree[i].Left = -1;
            _tree[i].Right = -1;
            _tree[i].Count = (long)1e15;
            _tree[i].Binary = false;
        }
        for (int i = 0; i < _osz; i++)
        {
            _tree[i].Count = counts[i];
        }
        int leaf = _osz - 1;
        int node = _osz;
        Span<int> mini = stackalloc int[2];
        for (int i = _osz; i < 2 * _osz - 1; i++)
        {
            for (int j = 0; j < 2; j++)
            {
                if (leaf >= 0 && _tree[leaf].Count < _tree[node].Count)
                {
                    mini[j] = leaf--;
                }
                else
                {
                    mini[j] = node++;
                }
            }
            _tree[i].Left = mini[0];
            _tree[i].Right = mini[1];
            _tree[i].Count = _tree[mini[0]].Count + _tree[mini[1]].Count;
            _tree[mini[0]].Parent = i;
            _tree[mini[1]].Parent = i;
            _tree[mini[1]].Binary = true;
        }
        for (int i = 0; i < _osz; i++)
        {
            var path = new List<int>();
            var code = new List<bool>();
            int j = i;
            while (_tree[j].Parent != -1)
            {
                path.Add(_tree[j].Parent - _osz);
                code.Add(_tree[j].Binary);
                j = _tree[j].Parent;
            }
            _paths[i] = path;
            _codes[i] = code;
        }
    }

    public override float Forward(List<int> targets, int targetIndex, ModelState state, float lr, bool backprop)
    {
        float loss = 0f;
        int target = targets[targetIndex];
        List<bool> binaryCode = _codes[target];
        List<int> pathToRoot = _paths[target];
        for (int i = 0; i < pathToRoot.Count; i++)
        {
            loss += BinaryLogistic(pathToRoot[i], state, binaryCode[i], lr, backprop);
        }
        return loss;
    }

    public override void Predict(int k, float threshold, List<Prediction> heap, ModelState state)
    {
        Dfs(k, threshold, 2 * _osz - 2, 0f, heap, state.Hidden);
        heap.Sort(static (a, b) =>
        {
            int c = b.Score.CompareTo(a.Score);
            return c != 0 ? c : a.Label.CompareTo(b.Label);
        });
    }

    private void Dfs(int k, float threshold, int node, float score, List<Prediction> heap, float[] hidden)
    {
        if (score < StdLog(threshold))
        {
            return;
        }
        if (heap.Count == k && score < MinScore(heap))
        {
            return;
        }
        if (_tree[node].Left == -1 && _tree[node].Right == -1)
        {
            Insert(heap, k, new Prediction(score, node));
            return;
        }
        float f = Wo.DotRow(hidden, node - _osz);
        f = 1f / (1f + (float)Math.Exp(-f));
        Dfs(k, threshold, _tree[node].Left, score + StdLog(1f - f), heap, hidden);
        Dfs(k, threshold, _tree[node].Right, score + StdLog(f), heap, hidden);
    }

    private static float MinScore(List<Prediction> heap)
    {
        float min = float.MaxValue;
        foreach (Prediction p in heap)
        {
            if (p.Score < min)
            {
                min = p.Score;
            }
        }
        return min;
    }

    private static void Insert(List<Prediction> heap, int k, Prediction p)
    {
        heap.Add(p);
        if (heap.Count > k)
        {
            int minIdx = 0;
            for (int i = 1; i < heap.Count; i++)
            {
                if (heap[i].Score < heap[minIdx].Score)
                {
                    minIdx = i;
                }
            }
            heap.RemoveAt(minIdx);
        }
    }
}

internal sealed class SoftmaxLoss : Loss
{
    public SoftmaxLoss(Matrix wo) : base(wo) { }

    public override void ComputeOutput(ModelState state)
    {
        float[] output = state.Output;
        ReadOnlySpan<float> hidden = state.Hidden;
        int osz = output.Length;
        for (int i = 0; i < osz; i++)
        {
            output[i] = Wo.DotRow(hidden, i);
        }
        float max = output[0];
        for (int i = 1; i < osz; i++)
        {
            max = Math.Max(output[i], max);
        }
        float z = 0f;
        for (int i = 0; i < osz; i++)
        {
            output[i] = (float)Math.Exp(output[i] - max);
            z += output[i];
        }
        for (int i = 0; i < osz; i++)
        {
            output[i] /= z;
        }
    }

    public override float Forward(List<int> targets, int targetIndex, ModelState state, float lr, bool backprop)
    {
        ComputeOutput(state);
        int target = targets[targetIndex];
        if (backprop)
        {
            int osz = state.Output.Length;
            for (int i = 0; i < osz; i++)
            {
                float label = i == target ? 1.0f : 0.0f;
                float alpha = lr * (label - state.Output[i]);
                Wo.AddRowToVector(state.Grad, i, alpha);
                Wo.AddVectorToRow(state.Hidden, i, alpha);
            }
        }
        return -Log(state.Output[target]);
    }
}
