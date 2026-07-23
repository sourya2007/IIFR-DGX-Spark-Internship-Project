import re
from pathlib import Path

import torch
from torch.utils.data import Dataset

PAD, BOS, EOS, UNK = 0, 1, 2, 3

_WORD_RE = re.compile(r"\S+|\n")


def build_vocab(texts, min_freq=2):
    counts = {}
    for text in texts:
        for t in _WORD_RE.findall(text):
            counts[t] = counts.get(t, 0) + 1
    sorted_words = sorted(w for w, c in counts.items() if c >= min_freq)
    word2idx = {w: i + 4 for i, w in enumerate(sorted_words)}
    idx2word = {i + 4: w for i, w in enumerate(sorted_words)}
    return word2idx, idx2word


def vocab_size(word2idx):
    return len(word2idx) + 4


def encode(text, word2idx):
    ids = [BOS]
    for t in _WORD_RE.findall(text):
        ids.append(word2idx.get(t, UNK))
    ids.append(EOS)
    return torch.tensor(ids, dtype=torch.long)


def encode_no_eos(text, word2idx):
    ids = [BOS]
    for t in _WORD_RE.findall(text):
        ids.append(word2idx.get(t, UNK))
    return torch.tensor(ids, dtype=torch.long)


def decode(ids, idx2word):
    out = []
    for i in ids:
        if i == EOS:
            break
        if i in (PAD, BOS):
            continue
        w = idx2word.get(i, "<?>")
        out.append(w)
    s = " ".join(out)
    s = s.replace("\n ", "\n")
    s = s.replace(" \n", "\n")
    return s.strip()


def save_vocab(word2idx, idx2word, path):
    data = {"w2i": word2idx, "i2w": {str(k): v for k, v in idx2word.items()}}
    torch.save(data, path)


def load_vocab(path):
    data = torch.load(path, weights_only=True)
    return data["w2i"], {int(k): v for k, v in data["i2w"].items()}


class TextDataset(Dataset):
    def __init__(self, filepath, max_seq, word2idx):
        self.max_seq = max_seq
        raw = Path(filepath).read_text(encoding="utf-8", errors="replace")
        raw = raw.encode("ascii", "replace").decode("ascii")
        ids = encode(raw, word2idx)
        self.seqs = []
        for i in range(0, len(ids) - 1, max_seq - 1):
            seg = ids[i : i + max_seq]
            if len(seg) >= 20:
                self.seqs.append(seg)

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        x = self.seqs[idx]
        pad = self.max_seq - len(x)
        x = torch.cat([x, torch.full((pad,), PAD, dtype=torch.long)])
        y = torch.cat([x[1:], torch.tensor([PAD])])
        y[x == PAD] = -100
        return x, y


def search_passages(query, full_text, top_k=3, window=400):
    query_words = set(w.lower() for w in _WORD_RE.findall(query) if len(w) > 2)
    if not query_words:
        return []
    chunks = []
    for i in range(0, len(full_text) - window + 1, window // 2):
        chunks.append(full_text[i : i + window])
    scored = []
    for chunk in chunks:
        chunk_words = set(w.lower() for w in _WORD_RE.findall(chunk) if len(w) > 2)
        score = len(query_words & chunk_words) / len(query_words)
        scored.append((score, chunk))
    scored.sort(key=lambda x: -x[0])
    return [c for s, c in scored[:top_k] if s > 0.05]
