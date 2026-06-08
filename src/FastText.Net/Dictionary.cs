// Ported from Facebook fastText (src/dictionary.h, src/dictionary.cc).
// See THIRD-PARTY-NOTICES.md. Original: Copyright (c) Facebook, Inc. (MIT License).

using System.Buffers;
using System.Text;

namespace FastTextNet;

internal enum EntryType : byte { Word = 0, Label = 1 }

internal struct Entry
{
    public byte[] Word;
    public long Count;
    public EntryType Type;
    public int[] Subwords;
}

/// <summary>
/// fastText dictionary. Operates on raw UTF-8 bytes to remain bit-compatible with the
/// reference implementation's FNV hash and subword tokenization (which both treat words
/// as byte sequences, not Unicode code points).
/// </summary>
internal sealed class Dictionary
{
    private const int MaxLineSize = 1024;
    private const int MaxVocabSize = 30000000;

    private static readonly byte[] Eos = "</s>"u8.ToArray();
    private const byte Bow = (byte)'<';
    private const byte Eow = (byte)'>';

    private readonly Args _args;
    private readonly byte[] _label;

    private Entry[] _words = Array.Empty<Entry>();
    private int[] _word2int = Array.Empty<int>();
    private Dictionary<int, int>? _pruneIdx;
    private long _pruneIdxSize = -1;

    private int _size;
    private int _nwords;
    private int _nlabels;
    private long _ntokens;

    private float[] _pdiscard = Array.Empty<float>();

    public Dictionary(Args args)
    {
        _args = args;
        _label = Encoding.ASCII.GetBytes(args.Label);
    }

    public int NWords => _nwords;
    public int NLabels => _nlabels;
    public long NTokens => _ntokens;

    public bool IsPruned => _pruneIdxSize >= 0;

    public EntryType GetEntryType(int id) => _words[id].Type;

    public string GetWord(int id) => Encoding.UTF8.GetString(_words[id].Word);

    public byte[] GetWordBytes(int id) => _words[id].Word;

    /// <summary>Subword ids for an arbitrary word (BOW/EOW bracketed), mirroring fastText.</summary>
    public List<int> GetSubwords(ReadOnlySpan<byte> word)
    {
        int i = GetId(word);
        var ngrams = new List<int>();
        if (i >= 0)
        {
            return new List<int>(_words[i].Subwords);
        }
        if (!word.SequenceEqual(Eos))
        {
            ComputeSubwords(Bracket(word), ngrams);
        }
        return ngrams;
    }

    public void GetSubwords(ReadOnlySpan<byte> word, List<int> ngrams, List<string> substrings)
    {
        int i = GetId(word);
        ngrams.Clear();
        substrings.Clear();
        if (i >= 0)
        {
            ngrams.Add(i);
            substrings.Add(Encoding.UTF8.GetString(_words[i].Word));
        }
        if (!word.SequenceEqual(Eos))
        {
            ComputeSubwords(Bracket(word), ngrams, substrings);
        }
    }

    public int GetSubwordId(ReadOnlySpan<byte> subword) =>
        _nwords + (int)(Hash(subword) % (uint)_args.Bucket);

    private static byte[] Bracket(ReadOnlySpan<byte> word)
    {
        byte[] bracketed = new byte[word.Length + 2];
        bracketed[0] = Bow;
        word.CopyTo(bracketed.AsSpan(1));
        bracketed[^1] = Eow;
        return bracketed;
    }


    public IReadOnlyList<long> GetCounts(EntryType type)
    {
        var counts = new List<long>();
        foreach (Entry e in _words)
        {
            if (e.Type == type)
            {
                counts.Add(e.Count);
            }
        }
        return counts;
    }

    // FNV-1a using the (historically signed-char) byte cast that fastText relies on.
    public static uint Hash(ReadOnlySpan<byte> str)
    {
        uint h = 2166136261;
        for (int i = 0; i < str.Length; i++)
        {
            h ^= (uint)(sbyte)str[i];
            h *= 16777619;
        }
        return h;
    }

    private int Find(ReadOnlySpan<byte> w) => Find(w, Hash(w));

    private int Find(ReadOnlySpan<byte> w, uint h)
    {
        int size = _word2int.Length;
        int id = (int)(h % (uint)size);
        while (_word2int[id] != -1 && !w.SequenceEqual(_words[_word2int[id]].Word))
        {
            id = (id + 1) % size;
        }
        return id;
    }

    public int GetId(ReadOnlySpan<byte> w) => _word2int[Find(w)];

    private int GetId(ReadOnlySpan<byte> w, uint h) => _word2int[Find(w, h)];

    private EntryType GetType(int id) => _words[id].Type;

    private EntryType GetType(ReadOnlySpan<byte> w) =>
        StartsWith(w, _label) ? EntryType.Label : EntryType.Word;

    private static bool StartsWith(ReadOnlySpan<byte> w, byte[] prefix) =>
        w.Length >= prefix.Length && w[..prefix.Length].SequenceEqual(prefix);

    public string GetLabel(int lid)
    {
        if (lid < 0 || lid >= _nlabels)
        {
            throw new ArgumentOutOfRangeException(nameof(lid));
        }
        return Encoding.UTF8.GetString(_words[lid + _nwords].Word);
    }

    private void ComputeSubwords(ReadOnlySpan<byte> word, List<int> ngrams, List<string>? substrings = null)
    {
        int n = word.Length;
        int minn = _args.Minn, maxn = _args.Maxn, bucket = _args.Bucket;
        for (int i = 0; i < n; i++)
        {
            if ((word[i] & 0xC0) == 0x80)
            {
                continue;
            }
            int j = i;
            for (int len = 1; j < n && len <= maxn; len++)
            {
                j++;
                while (j < n && (word[j] & 0xC0) == 0x80)
                {
                    j++;
                }
                if (len >= minn && !(len == 1 && (i == 0 || j == n)))
                {
                    int hh = (int)(Hash(word.Slice(i, j - i)) % (uint)bucket);
                    PushHash(ngrams, hh);
                    substrings?.Add(Encoding.UTF8.GetString(word.Slice(i, j - i)));
                }
            }
        }
    }

    private void PushHash(List<int> hashes, int id)
    {
        if (_pruneIdxSize == 0 || id < 0)
        {
            return;
        }
        if (_pruneIdxSize > 0)
        {
            if (_pruneIdx!.TryGetValue(id, out int mapped))
            {
                id = mapped;
            }
            else
            {
                return;
            }
        }
        hashes.Add(_nwords + id);
    }

    private void AddSubwords(List<int> line, ReadOnlySpan<byte> token, int wid)
    {
        if (wid < 0)
        {
            if (!token.SequenceEqual(Eos))
            {
                Span<byte> stack = token.Length <= 256 ? stackalloc byte[token.Length + 2] : default;
                byte[]? rented = null;
                Span<byte> concat = stack.IsEmpty
                    ? (rented = ArrayPool<byte>.Shared.Rent(token.Length + 2)).AsSpan(0, token.Length + 2)
                    : stack;
                concat[0] = Bow;
                token.CopyTo(concat[1..]);
                concat[token.Length + 1] = Eow;
                ComputeSubwords(concat, line);
                if (rented is not null)
                {
                    ArrayPool<byte>.Shared.Return(rented);
                }
            }
        }
        else if (_args.Maxn <= 0)
        {
            line.Add(wid);
        }
        else
        {
            line.AddRange(_words[wid].Subwords);
        }
    }

    private void AddWordNgrams(List<int> line, List<int> hashes, int n)
    {
        int bucket = _args.Bucket;
        for (int i = 0; i < hashes.Count; i++)
        {
            // Sign-extend through int32 exactly as the reference does (vector<int32_t>).
            ulong h = (ulong)(long)hashes[i];
            for (int j = i + 1; j < hashes.Count && j < i + n; j++)
            {
                h = h * 116049371 + (ulong)(long)hashes[j];
                PushHash(line, (int)(h % (uint)bucket));
            }
        }
    }

    /// <summary>
    /// Tokenizes a whitespace-delimited UTF-8 line into input feature ids, matching
    /// fastText's predictLine/getLine tokenization. An end-of-sentence token (<c>&lt;/s&gt;</c>)
    /// is appended, mirroring the reference CLI/Python behavior of terminating the line.
    /// </summary>
    public void GetLine(ReadOnlySpan<byte> input, List<int> words)
    {
        words.Clear();
        var wordHashes = new List<int>();
        int pos = 0;
        while (TryReadWord(input, ref pos, out ReadOnlySpan<byte> token))
        {
            ProcessToken(token, words, wordHashes, null);
        }
        ProcessToken(Eos, words, wordHashes, null);
        AddWordNgrams(words, wordHashes, _args.WordNgrams);
    }

    /// <summary>
    /// Tokenizes a line into input feature ids and label ids, mirroring fastText's
    /// supervised <c>getLine(in, words, labels)</c> used by <c>test</c>/<c>predict</c>.
    /// </summary>
    public int GetLine(ReadOnlySpan<byte> input, List<int> words, List<int> labels)
    {
        words.Clear();
        labels.Clear();
        var wordHashes = new List<int>();
        int ntokens = 0;
        int pos = 0;
        while (TryReadWord(input, ref pos, out ReadOnlySpan<byte> token))
        {
            ProcessToken(token, words, wordHashes, labels);
            ntokens++;
        }
        ProcessToken(Eos, words, wordHashes, labels);
        ntokens++;
        AddWordNgrams(words, wordHashes, _args.WordNgrams);
        return ntokens;
    }

    private void ProcessToken(ReadOnlySpan<byte> token, List<int> words, List<int> wordHashes, List<int>? labels)
    {
        uint h = Hash(token);
        int wid = GetId(token, h);
        EntryType type = wid < 0 ? GetType(token) : GetType(wid);
        if (type == EntryType.Word)
        {
            AddSubwords(words, token, wid);
            wordHashes.Add((int)h);
        }
        else if (type == EntryType.Label && wid >= 0)
        {
            labels?.Add(wid - _nwords);
        }
    }

    private static bool IsSpace(byte b) =>
        b is (byte)' ' or (byte)'\n' or (byte)'\r' or (byte)'\t' or 0x0B or 0x0C or 0;

    private static bool TryReadWord(ReadOnlySpan<byte> input, ref int pos, out ReadOnlySpan<byte> word)
    {
        while (pos < input.Length && IsSpace(input[pos]))
        {
            pos++;
        }
        if (pos >= input.Length)
        {
            word = default;
            return false;
        }
        int start = pos;
        while (pos < input.Length && !IsSpace(input[pos]))
        {
            pos++;
        }
        word = input.Slice(start, pos - start);
        return true;
    }

    public int[] GetSubwordsById(int id) => _words[id].Subwords;

    public bool Discard(int id, float rand)
    {
        if (_args.Model == ModelName.Sup)
        {
            return false;
        }
        return rand > _pdiscard[id];
    }

    /// <summary>
    /// Builds the vocabulary from a training corpus, mirroring fastText's readFromFile:
    /// newlines are end-of-sentence tokens, rare entries are pruned by minCount, and the
    /// discard table and subword ngrams are initialized.
    /// </summary>
    public void ReadFromFile(ReadOnlySpan<byte> data)
    {
        _word2int = new int[MaxVocabSize];
        Array.Fill(_word2int, -1);
        var words = new List<Entry>();
        _size = 0;
        _nwords = 0;
        _nlabels = 0;
        _ntokens = 0;

        long minThreshold = 1;
        int pos = 0;
        while (ReadWordBuild(data, ref pos, out ReadOnlySpan<byte> token))
        {
            AddBuild(words, token);
            if (_size > 0.75 * MaxVocabSize)
            {
                minThreshold++;
                ThresholdBuild(words, minThreshold, minThreshold);
            }
        }
        ThresholdBuild(words, _args.MinCount, _args.MinCountLabel);

        _words = words.ToArray();
        InitTableDiscard();
        InitNgrams();

        if (_size == 0)
        {
            throw new InvalidOperationException("Empty vocabulary. Try a smaller minCount value.");
        }
    }

    private void AddBuild(List<Entry> words, ReadOnlySpan<byte> w)
    {
        int h = FindBuild(words, w, Hash(w));
        _ntokens++;
        if (_word2int[h] == -1)
        {
            words.Add(new Entry
            {
                Word = w.ToArray(),
                Count = 1,
                Type = GetType(w),
                Subwords = Array.Empty<int>(),
            });
            _word2int[h] = _size++;
        }
        else
        {
            Entry e = words[_word2int[h]];
            e.Count++;
            words[_word2int[h]] = e;
        }
    }

    private int FindBuild(List<Entry> words, ReadOnlySpan<byte> w, uint h)
    {
        int size = _word2int.Length;
        int id = (int)(h % (uint)size);
        while (_word2int[id] != -1 && !w.SequenceEqual(words[_word2int[id]].Word))
        {
            id = (id + 1) % size;
        }
        return id;
    }

    private void ThresholdBuild(List<Entry> words, long t, long tl)
    {
        words.Sort((e1, e2) =>
            e1.Type != e2.Type
                ? ((byte)e1.Type).CompareTo((byte)e2.Type)
                : e2.Count.CompareTo(e1.Count));
        words.RemoveAll(e =>
            (e.Type == EntryType.Word && e.Count < t) ||
            (e.Type == EntryType.Label && e.Count < tl));

        _size = 0;
        _nwords = 0;
        _nlabels = 0;
        Array.Fill(_word2int, -1);
        foreach (Entry e in words)
        {
            int h = FindBuild(words, e.Word, Hash(e.Word));
            _word2int[h] = _size++;
            if (e.Type == EntryType.Word)
            {
                _nwords++;
            }
            else
            {
                _nlabels++;
            }
        }
    }

    private void InitTableDiscard()
    {
        _pdiscard = new float[_size];
        for (int i = 0; i < _size; i++)
        {
            float f = (float)_words[i].Count / _ntokens;
            _pdiscard[i] = (float)(Math.Sqrt(_args.T / f) + _args.T / f);
        }
    }

    // Mirrors fastText's readWord: whitespace-delimited, with '\n' yielding the EOS token.
    private bool ReadWordBuild(ReadOnlySpan<byte> data, ref int pos, out ReadOnlySpan<byte> word)
    {
        int wordStart = -1;
        int wlen = 0;
        while (pos < data.Length)
        {
            byte c = data[pos++];
            if (IsSpace(c))
            {
                if (wlen == 0)
                {
                    if (c == (byte)'\n')
                    {
                        word = Eos;
                        return true;
                    }
                    continue;
                }
                if (c == (byte)'\n')
                {
                    pos--;
                }
                word = data.Slice(wordStart, wlen);
                return true;
            }
            if (wlen == 0)
            {
                wordStart = pos - 1;
            }
            wlen++;
        }
        if (wlen > 0)
        {
            word = data.Slice(wordStart, wlen);
            return true;
        }
        word = default;
        return false;
    }

    /// <summary>
    /// Reads one training line of word ids for the unsupervised models, applying
    /// frequency-based subsampling (discard). Returns the number of tokens consumed.
    /// </summary>
    public int GetLine(ReadOnlySpan<byte> lineBytes, List<int> words, ref MinstdRand rng)
    {
        words.Clear();
        int ntokens = 0;
        int pos = 0;
        while (TryReadWord(lineBytes, ref pos, out ReadOnlySpan<byte> token))
        {
            uint h = Hash(token);
            int wid = GetId(token, h);
            if (wid < 0)
            {
                continue;
            }
            ntokens++;
            if (GetType(wid) == EntryType.Word && !Discard(wid, rng.NextFloat()))
            {
                words.Add(wid);
            }
            if (ntokens > MaxLineSize)
            {
                return ntokens;
            }
        }
        int eosId = GetId(Eos);
        if (eosId >= 0)
        {
            ntokens++;
            if (!Discard(eosId, rng.NextFloat()))
            {
                words.Add(eosId);
            }
        }
        return ntokens;
    }

    public void Load(BinaryReader reader)
    {
        _size = reader.ReadInt32();
        _nwords = reader.ReadInt32();
        _nlabels = reader.ReadInt32();
        _ntokens = reader.ReadInt64();
        _pruneIdxSize = reader.ReadInt64();

        _words = new Entry[_size];
        var wordBytes = new List<byte>(64);
        for (int i = 0; i < _size; i++)
        {
            wordBytes.Clear();
            byte c;
            while ((c = reader.ReadByte()) != 0)
            {
                wordBytes.Add(c);
            }
            _words[i] = new Entry
            {
                Word = wordBytes.ToArray(),
                Count = reader.ReadInt64(),
                Type = (EntryType)reader.ReadByte(),
                Subwords = Array.Empty<int>(),
            };
        }

        _pruneIdx = new Dictionary<int, int>();
        for (long i = 0; i < _pruneIdxSize; i++)
        {
            int first = reader.ReadInt32();
            int second = reader.ReadInt32();
            _pruneIdx[first] = second;
        }

        InitNgrams();

        int word2intsize = (int)Math.Ceiling(_size / 0.7);
        _word2int = new int[word2intsize];
        Array.Fill(_word2int, -1);
        for (int i = 0; i < _size; i++)
        {
            _word2int[Find(_words[i].Word)] = i;
        }
    }

    public int EosId => GetId(Eos);

    /// <summary>
    /// Prunes the dictionary to the given input-matrix row indices, mirroring fastText's
    /// quantization cutoff. Rewrites <paramref name="idx"/> in place to the retained rows
    /// (sorted words followed by retained ngram buckets).
    /// </summary>
    public void Prune(List<int> idx)
    {
        var words = new List<int>();
        var ngrams = new List<int>();
        foreach (int i in idx)
        {
            if (i < _nwords)
            {
                words.Add(i);
            }
            else
            {
                ngrams.Add(i);
            }
        }
        words.Sort();
        idx.Clear();
        idx.AddRange(words);

        _pruneIdx ??= new Dictionary<int, int>();
        if (ngrams.Count != 0)
        {
            int j = 0;
            foreach (int ngram in ngrams)
            {
                _pruneIdx[ngram - _nwords] = j;
                j++;
            }
            idx.AddRange(ngrams);
        }
        _pruneIdxSize = _pruneIdx.Count;

        var compacted = new Entry[_words.Length];
        int next = 0;
        for (int i = 0; i < _words.Length; i++)
        {
            if (_words[i].Type == EntryType.Label || (next < words.Count && words[next] == i))
            {
                compacted[next] = _words[i];
                next++;
            }
        }
        _nwords = words.Count;
        _size = _nwords + _nlabels;
        _words = compacted[..next];

        Array.Fill(_word2int, -1);
        for (int i = 0; i < _size; i++)
        {
            _word2int[Find(_words[i].Word)] = i;
        }
        InitNgrams();
    }

    public void Save(BinaryWriter writer)
    {
        writer.Write(_size);
        writer.Write(_nwords);
        writer.Write(_nlabels);
        writer.Write(_ntokens);
        writer.Write(_pruneIdxSize);
        for (int i = 0; i < _size; i++)
        {
            writer.Write(_words[i].Word);
            writer.Write((byte)0);
            writer.Write(_words[i].Count);
            writer.Write((byte)_words[i].Type);
        }
        if (_pruneIdx is not null)
        {
            foreach (KeyValuePair<int, int> pair in _pruneIdx)
            {
                writer.Write(pair.Key);
                writer.Write(pair.Value);
            }
        }
    }

    private void InitNgrams()
    {
        var subwords = new List<int>();
        for (int i = 0; i < _size; i++)
        {
            byte[] w = _words[i].Word;
            subwords.Clear();
            subwords.Add(i);
            if (!w.AsSpan().SequenceEqual(Eos))
            {
                byte[] bracketed = new byte[w.Length + 2];
                bracketed[0] = Bow;
                w.CopyTo(bracketed, 1);
                bracketed[^1] = Eow;
                ComputeSubwords(bracketed, subwords);
            }
            _words[i].Subwords = subwords.ToArray();
        }
    }
}
