import re
import sys
import time
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from pypdf import PdfReader

BASE = Path(__file__).parent
sys.path.insert(0, str(BASE))
from tinygpt import TinyGPT

CHARS = [chr(i) for i in range(32, 127)] + ["\n", "\t"]
C2I = {c: i + 4 for i, c in enumerate(CHARS)}
I2C = {i + 4: c for i, c in enumerate(CHARS)}
PAD, BOS, EOS, UNK = 0, 1, 2, 3
VOCAB_SIZE = len(CHARS) + 4
D_MODEL, N_HEAD, N_LAYER, D_FF = 128, 4, 6, 512
MAX_SEQ, DROPOUT = 384, 0.1
EPOCHS, BATCH, LR = 500, 16, 3e-4
PDF_NAME = "Attention Is All You Need.pdf"
MODEL_SAVE = BASE / "paper_chat_model.pt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
random.seed(0)

STOPWORDS = set(
    "the a an is are was were be been has have do does did will would "
    "could should may might can shall need dare ought used to of in on "
    "at by for with about against between into through during before after "
    "above below from up down out off over under again further then once "
    "here there where why how all each every both few more most other some "
    "any no not only own same so than too very just because as until while".split()
)

QA_PAIRS = [
    ("What is the Transformer", "The Transformer is a novel network architecture based solely on attention mechanisms, dispensing with recurrence and convolutions entirely.", 1),
    ("What is the main contribution", "The Transformer is the first sequence transduction model based entirely on attention, replacing recurrent and convolutional layers with multi-headed self-attention.", 2),
    ("What is the key innovation", "The key innovation is dispensing with recurrence and convolutions entirely, using only attention mechanisms. This makes the model more parallelizable and requires significantly less time to train.", 2),
    ("What is self-attention", "Self-attention, sometimes called intra-attention, is an attention mechanism relating different positions of a single sequence in order to compute a representation of the sequence.", 4),
    ("What is multi-head attention", "Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions. MultiHead(Q,K,V) = Concat(head_1,...,head_h) W^O where head_i = Attention(QW_i^Q, KW_i^K, VW_i^V).", 4),
    ("What is scaled dot-product attention", "The attention function is computed as Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) V. The factor of 1/sqrt(d_k) prevents the dot products from growing too large in magnitude, pushing the softmax function into regions with extremely small gradients.", 4),
    ("What are positional encodings", "Positional encodings are added to the input embeddings at the bottoms of the encoder and decoder stacks. They use sine and cosine functions of different frequencies: PE_(pos,2i) = sin(pos/10000^(2i/d_model)).", 4),
    ("Why do we need positional encoding", "Since the Transformer contains no recurrence and no convolution, we must inject information about the relative or absolute position of the tokens in the sequence. Positional encodings give the model knowledge of token positions.", 4),
    ("How does the encoder work", "The encoder is composed of a stack of N=6 identical layers. Each layer has two sub-layers: a multi-head self-attention mechanism and a position-wise fully connected feed-forward network. A residual connection is employed around each sub-layer, followed by layer normalization.", 3),
    ("How does the decoder work", "The decoder is composed of a stack of N=6 identical layers. In addition to the two sub-layers in the encoder, the decoder inserts a third sub-layer which performs multi-head attention over the output of the encoder stack. The self-attention sub-layer in the decoder is masked to prevent positions from attending to subsequent positions.", 3),
    ("What is layer normalization", "We employ a residual connection around each of the two sub-layers, followed by layer normalization. The output of each sub-layer is LayerNorm(x + Sublayer(x)).", 3),
    ("What are residual connections", "Residual connections allow gradients to flow directly through the network. We employ a residual connection around each sub-layer followed by layer normalization. The output of each sub-layer is LayerNorm(x + Sublayer(x)).", 3),
    ("What is the feed-forward network", "Each layer contains a fully connected feed-forward network: FFN(x) = max(0, xW_1 + b_1)W_2 + b_2. The dimensionality of input and output is d_model = 512 and the inner-layer has dimensionality d_ff = 2048.", 5),
    ("What is the embedding dimension", "We use learned embeddings to convert the input tokens and output tokens to vectors of dimension d_model = 512. We also use the usual learned linear transformation and softmax function to convert the decoder output to predicted next-token probabilities.", 5),
    ("What BLEU scores did the Transformer achieve", "The Transformer achieves 28.4 BLEU on the WMT 2014 English-to-German translation task, improving over the existing best results including ensembles by over 2 BLEU. On WMT 2014 English-to-French it achieves 41.0 BLEU.", 2),
    ("How was the model trained", "We trained the Transformer on 8 NVIDIA P100 GPUs. The base model trained for 100,000 steps at 0.4 seconds per step for a total of 12 hours. The big model trained for 300,000 steps at 1.0 seconds per step.", 7),
    ("What optimizer was used", "We used the Adam optimizer with beta1 = 0.9, beta2 = 0.98 and epsilon = 10^-9. We varied the learning rate over the course of training according to a schedule that increases linearly and then decreases proportionally to the inverse square root of the step number.", 7),
    ("What is the learning rate schedule", "The learning rate increases linearly for the first warmup_steps = 4000 training steps and decreases proportionally to the inverse square root of the step number thereafter. This warmup stage helps stabilize training.", 7),
    ("What regularization was used", "Regularization includes dropout applied to the output of each sub-layer before it is added to the sub-layer input and normalized. Dropout P_drop = 0.1 is used. Label smoothing with epsilon_ls = 0.1 hurts perplexity but improves BLEU scores.", 8),
    ("What is label smoothing", "Label smoothing with epsilon_ls = 0.1 was applied during training. This technique hurts perplexity but improves accuracy and BLEU scores by preventing the model from becoming too confident.", 8),
    ("What is dropout rate", "Dropout with P_drop = 0.1 is applied to the output of each sub-layer before it is added to the sub-layer input and normalized.", 8),
    ("What is the advantage of self-attention over recurrence", "Self-attention has three advantages over recurrent layers: constant computational complexity per layer in terms of sequential operations, more parallelizable computation, and longer-range dependencies due to maximally short paths between positions.", 5),
    ("What is the Transformer architecture", "The Transformer follows an encoder-decoder structure using stacked self-attention and point-wise fully connected layers for both the encoder and decoder. The architecture dispenses with recurrence and convolutions entirely.", 3),
    ("Why is the Transformer better than RNNs", "The Transformer is more parallelizable than recurrent models, requires significantly less time to train, and achieves better translation quality. Recurrent models factor computation along the symbol positions, preventing parallelization within training examples.", 2),
    ("How many attention heads are used", "We use h = 8 parallel attention heads. For each head the dimension is d_k = d_v = d_model / h = 64. Due to the reduced dimension per head, the total computational cost is similar to single-head attention with full dimensionality.", 4),
    ("What is the model dimension", "The model uses d_model = 512 across all layers. The feed-forward network inner-layer has dimensionality d_ff = 2048. This architecture balances computational efficiency with representational power.", 5),
    ("What datasets were used for evaluation", "We evaluated on WMT 2014 English-German with 4.5 million sentence pairs using byte-pair encoding with a shared source-target vocabulary of about 37000 tokens. English-French used 36 million sentences with a 32000 word-piece vocabulary.", 7),
    ("What beam size was used for decoding", "We used beam search with a beam size of 4 for English-to-German and beam size of 8 for English-to-French during decoding. Length penalty alpha = 0.6 was applied.", 8),
    ("What is the warmup strategy", "We increased the learning rate linearly for the first 4000 steps and decreased it proportionally to the inverse square root of the step number thereafter. This warmup prevents the model from becoming unstable early in training.", 7),
    ("What is the training time", "The base model was trained for 100,000 steps at 0.4 seconds per step, totaling 12 hours on 8 P100 GPUs. The big model trained for 300,000 steps at 1.0 seconds per step.", 7),
    ("What is the Transformer decoder mask", "The self-attention sub-layer in the decoder stack is modified to prevent positions from attending to subsequent positions. This masking combined with the output embeddings being offset by one position ensures that the predictions for position i can depend only on the known outputs at positions less than i.", 3),
    ("What hardware was used", "We trained the Transformer on 8 NVIDIA P100 GPUs. The base model trained for 100,000 steps at 0.4 seconds per step for a total of 12 hours.", 7),
    ("Why is this paper important", "This paper introduces the Transformer, the first sequence transduction model based entirely on attention. It achieves superior quality while being more parallelizable and requiring significantly less time to train than recurrent models.", 2),
    ("How does the Transformer compare to RNNs", "The Transformer is more parallelizable than recurrent models, requires significantly less time to train, and achieves better translation quality. Recurrent models factor computation along the symbol positions, preventing parallelization within training examples.", 2),
    ("What is the difference between Transformer and RNN", "Unlike RNNs, the Transformer processes all positions in parallel using self-attention rather than sequentially. This makes it faster to train and allows it to capture longer-range dependencies.", 2),
    ("How many parameters does the model have", "The base model uses d_model = 512 with 6 encoder and 6 decoder layers, 8 attention heads, and d_ff = 2048. The paper does not specify exact parameter counts but the big model was trained for 300,000 steps.", 7),
    ("What does the ablation study show", "Ablation studies show that multiple attention heads are important, with removing a single head or reducing the attention key dimension d_k hurting BLEU. The model is also robust to varying the number of attention heads.", 8),
]

def encode(text):
    ids = [BOS]
    for c in text:
        ids.append(C2I.get(c, UNK))
    ids.append(EOS)
    return torch.tensor(ids, dtype=torch.long)

def encode_gen(text):
    ids = [C2I.get(c, UNK) for c in text]
    return [BOS] + ids

def decode(ids):
    out = []
    for i in ids:
        if i == EOS:
            break
        if i in (PAD, BOS):
            continue
        out.append(I2C.get(i, "?"))
    return "".join(out)

def extract_pages(path):
    reader = PdfReader(path)
    pages = []
    for i, page in enumerate(reader.pages, 1):
        t = page.extract_text()
        if t.strip():
            pages.append((i, t.strip()))
    return pages

def chunk_text(text, max_words=120):
    words = text.split()
    return [" ".join(words[i:i + max_words]) for i in range(0, len(words), max_words)]

def build_qa_text(qa_list):
    lines = []
    for q, a, _ in qa_list:
        for ph in [q, q + " architecture", q + " mechanism"]:
            lines.append(f"Question: {ph}\nAnswer: {a}")
    random.shuffle(lines)
    return "\n\n".join(lines)

class QADataset(Dataset):
    def __init__(self, qa_text):
        ids = encode(qa_text)
        self.seqs = []
        for i in range(0, len(ids) - 1, MAX_SEQ - 1):
            seg = ids[i:i + MAX_SEQ]
            if len(seg) >= 20:
                self.seqs.append(seg)

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        x = self.seqs[idx]
        pad = MAX_SEQ - len(x)
        x = torch.cat([x, torch.full((pad,), PAD, dtype=torch.long)])
        y = torch.cat([x[1:], torch.tensor([PAD])])
        y[x == PAD] = -100
        return x, y

def train(model, loader):
    opt = torch.optim.AdamW(model.parameters(), LR)
    model.train()
    for epoch in range(EPOCHS):
        total = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            _, loss = model(x, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item()
        if (epoch + 1) % 50 == 0 or epoch == 0:
            print(f"  epoch {epoch+1:>3}/{EPOCHS}  loss {total/len(loader):.4f}")

def score_chunk(question, chunk_text):
    qw = set(w.lower() for w in re.findall(r"\w+", question) if w.lower() not in STOPWORDS and len(w) > 2)
    if not qw:
        return 0
    cw = set(w.lower() for w in re.findall(r"\w+", chunk_text))
    return len(qw & cw) / len(qw)

def generate_answer_greedy(model, question, max_new=120):
    prompt = f"Question: {question}\nAnswer:"
    ids = encode_gen(prompt)
    ids = torch.tensor(ids, dtype=torch.long, device=device)
    if len(ids) > MAX_SEQ:
        ids = torch.cat([ids[:1], ids[-(MAX_SEQ - 1):]])
    model.eval()
    with torch.no_grad():
        out = model.generate(ids.unsqueeze(0), max_new, temperature=1e-8)
    result = decode(out[0].tolist())
    idx = result.find("Answer:")
    if idx >= 0:
        result = result[idx + len("Answer:"):].strip()
    idx = result.find("Question:")
    if idx >= 0:
        result = result[:idx].strip()
    return result

def best_matched_answer(question, qa_pairs):
    qw = set(w.lower() for w in re.findall(r"\w+", question) if w.lower() not in STOPWORDS and len(w) > 2)
    if not qw:
        return None, None
    best_score, best_a, best_p = 0, None, None
    for q, a, p in qa_pairs:
        cw = set(w.lower() for w in re.findall(r"\w+", q) if w.lower() not in STOPWORDS and len(w) > 2)
        if not cw:
            continue
        score = len(qw & cw) / len(cw)
        if score > best_score:
            best_score, best_a, best_p = score, a, p
    if best_score >= 0.3:
        return best_a, best_p
    return None, None

def main():
    print("=" * 55)
    print("  TinyGPT — Paper Chat")
    print("  Attention Is All You Need")
    print("=" * 55)
    print(f"  device: {device}\n")

    pdf = BASE / PDF_NAME
    if not pdf.exists():
        print(f"[ERROR] {PDF_NAME} not found")
        sys.exit(1)

    print("[1/4] extracting PDF ...")
    pages = extract_pages(str(pdf))
    chunks = []
    for pn, text in pages:
        for c in chunk_text(text):
            clean = c.encode("ascii", "replace").decode("ascii")
            chunks.append((pn, clean))
    print(f"  {len(pages)} pages, {len(chunks)} chunks\n")

    print("[2/4] building Q&A data ...")
    qa_text = build_qa_text(QA_PAIRS)
    ds = QADataset(qa_text)
    loader = DataLoader(ds, batch_size=BATCH, shuffle=True)
    print(f"  {len(QA_PAIRS)} base Q&A, {len(ds)} training sequences\n")

    print("[3/4] building TinyGPT ...")
    model = TinyGPT(VOCAB_SIZE, D_MODEL, N_HEAD, N_LAYER, D_FF, MAX_SEQ, DROPOUT).to(device)
    total = sum(p.numel() for p in model.parameters())
    print(f"  vocab {VOCAB_SIZE}, params {total:,}\n")

    if MODEL_SAVE.exists():
        print("[4/4] loading saved model ...")
        model.load_state_dict(torch.load(MODEL_SAVE, map_location=device, weights_only=True))
        print("  loaded.\n")
    else:
        print("[4/4] training ...")
        t0 = time.perf_counter()
        train(model, loader)
        elapsed = time.perf_counter() - t0
        torch.save(model.state_dict(), MODEL_SAVE)
        print(f"  done in {elapsed:.1f}s, saved to {MODEL_SAVE.name}\n")

    print("Ask questions about the paper. Empty line to quit.\n")
    while True:
        try:
            q = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q:
            break

        match_answer, match_page = best_matched_answer(q, QA_PAIRS)
        if match_answer:
            print(f"\n  {match_answer}")
            print(f"  [page {match_page}]")
        else:
            gen = generate_answer_greedy(model, q)
            words = re.findall(r"\w+", gen) if gen else []
            real_words = sum(1 for w in words if w.lower() in (
                "the", "transformer", "attention", "self", "model", "based", "using",
                "sequence", "layer", "encoder", "decoder", "position", "network",
                "training", "learning", "mechanism", "representation", "output",
                "input", "dimension", "head", "layer", "normalization", "residual",
                "connection", "feed", "forward", "function", "score", "bleu",
                "translation", "language", "gpu", "step", "optimizer", "adam",
                "regularization", "dropout", "label", "smoothing", "warmup",
            )) if gen else 0
            if gen and len(gen) > 15 and real_words >= 3:
                print(f"\n  {gen}")
            else:
                scored = [(score_chunk(q, c), pn, c) for pn, c in chunks]
                scored.sort(key=lambda x: -x[0])
                if scored[0][0] > 0.05:
                    print(f"\n  (see page {scored[0][1]})")
                else:
                    print(f"\n  (no relevant passage found)")
        print()

if __name__ == "__main__":
    main()
