// Ported from Facebook fastText (src/model.h, src/model.cc, src/loss.h, src/loss.cc).
// See THIRD-PARTY-NOTICES.md. Original: Copyright (c) Facebook, Inc. (MIT License).

namespace FastTextNet;

internal sealed class ModelState
{
    public readonly float[] Hidden;
    public readonly float[] Output;

    public ModelState(int hiddenSize, int outputSize)
    {
        Hidden = new float[hiddenSize];
        Output = new float[outputSize];
    }
}

internal sealed class Model
{
    private readonly Matrix _wi;
    private readonly Matrix _wo;
    private readonly Loss _loss;

    public Model(Matrix wi, Matrix wo, Loss loss)
    {
        _wi = wi;
        _wo = wo;
        _loss = loss;
    }

    public int OutputSize => (int)_wo.Rows;

    public void Predict(List<int> input, int k, float threshold, List<Prediction> heap, ModelState state)
    {
        if (k <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(k), "k needs to be 1 or higher!");
        }
        _wi.AverageRowsToVector(state.Hidden, input);
        _loss.Predict(k, threshold, heap, state);
    }
}

internal readonly record struct Prediction(float Score, int Label);

internal abstract class Loss
{
    private const int SigmoidTableSize = 512;
    private const int MaxSigmoid = 8;

    private static readonly float[] SigmoidTable = BuildSigmoidTable();

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

    protected static float StdLog(float x) => (float)Math.Log((double)x + 1e-5);

    public abstract void ComputeOutput(ModelState state);

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
}

internal sealed class SigmoidLoss : Loss
{
    public SigmoidLoss(Matrix wo) : base(wo) { }

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

internal sealed class HierarchicalSoftmaxLoss : Loss
{
    private struct Node
    {
        public int Left;
        public int Right;
        public long Count;
        public bool Binary;
    }

    private readonly Node[] _tree;
    private readonly int _osz;

    public HierarchicalSoftmaxLoss(Matrix wo, IReadOnlyList<long> counts) : base(wo)
    {
        _osz = counts.Count;
        _tree = new Node[2 * _osz - 1];
        BuildTree(counts);
    }

    public override void ComputeOutput(ModelState state) =>
        throw new NotSupportedException("Hierarchical softmax uses tree traversal for prediction.");

    private void BuildTree(IReadOnlyList<long> counts)
    {
        var parent = new int[2 * _osz - 1];
        for (int i = 0; i < 2 * _osz - 1; i++)
        {
            parent[i] = -1;
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
            parent[mini[0]] = i;
            parent[mini[1]] = i;
            _tree[mini[1]].Binary = true;
        }
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
