"""Fine-tune ONE compact encoder on every task at once, then export it to ONNX for CPU inference.

The exported graph takes token ids and returns four things from a single forward pass:

    embedding   unit-norm sentence vector -> story clustering, novelty, analog retrieval
    sentiment   3-way logits (negative / neutral / positive) for the whole text
    event       logits over the event taxonomy
    target      3-way logits for the sentiment towards a *named entity* (text-pair input)

Why multi-task: the frozen-encoder probe (``scripts/select_encoder.py``) plateaus near 0.80
accuracy on news sentiment regardless of model size, so the encoder itself has to adapt to
financial language. Training all heads on one trunk keeps inference at one small model.

Why the distillation term: fine-tuning for classification normally wrecks an encoder's notion
of semantic similarity, which story clustering depends on. So the pooled vector used for
clustering is held close to the original model's vector (cosine loss against a frozen teacher),
while classification reads a different pooling of the same hidden states.

Needs the training environment (PyTorch + transformers); a GPU makes it minutes instead of an hour.

    python main.py finetune
"""

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from tremor.config import load_taxonomy
from tremor.nlp.preprocess import clean_text
from tremor.paths import DOCS_DIR, MODELS_DIR
from tremor.training.corpora import FIQA_NEUTRAL_BAND, SENTIMENT_LABELS, event_corpus, sentiment_corpus
from tremor.training.datasets import load_fiqa, load_sentfin, stable_split

MODEL_DIR = MODELS_DIR / "tremor-encoder"
DEV_FRACTION = 0.1  # of the training rows, held out for model selection (never the test split)


@dataclass
class FinetuneConfig:
    base_model: str = "BAAI/bge-small-en-v1.5"
    semantic_pooling: str = "cls"  # how the base model produces its sentence embedding
    max_length: int = 64
    pair_max_length: int = 80
    batch_size: int = 48
    epochs: int = 4
    lr: float = 5e-5
    warmup_fraction: float = 0.06
    weight_decay: float = 0.01
    distill_weight: float = 2.0
    seed: int = 13


# --------------------------------------------------------------------------- data
def _dev_split(df: pd.DataFrame) -> pd.DataFrame:
    """Carve a dev set out of the training rows with a stable hash (different salt from the test hash)."""
    df = df.copy()
    is_dev = df["text"].map(lambda t: stable_split("dev|" + t, DEV_FRACTION) == "test")
    df.loc[(df["split"] == "train") & is_dev, "split"] = "dev"
    return df


def target_corpus() -> pd.DataFrame:
    """(entity, text) pairs with the sentiment towards that entity. Columns: entity, text, label, split, dataset, n_entities."""
    sf = load_sentfin()
    sf["dataset"] = "sentfin"
    fq = load_fiqa()
    fq["label"] = fq["score"].map(lambda s: 0 if s <= -FIQA_NEUTRAL_BAND else (2 if s >= FIQA_NEUTRAL_BAND else 1))
    fq = fq.rename(columns={"target": "entity"})
    fq["dataset"], fq["n_entities"] = "fiqa", 1
    cols = ["entity", "text", "label", "split", "dataset", "n_entities"]
    df = pd.concat([sf[cols], fq[cols]], ignore_index=True)
    df["text"] = df["text"].map(clean_text)
    df = df[(df["text"].str.len() >= 8) & df["entity"].notna()]
    return df.reset_index(drop=True)


def class_weights(labels: np.ndarray, n_classes: int, power: float = 0.5) -> np.ndarray:
    counts = np.bincount(labels, minlength=n_classes).astype(np.float64)
    weights = (counts.max() / np.clip(counts, 1, None)) ** power
    return weights / weights.mean()


# --------------------------------------------------------------------------- model
def build_model(cfg: FinetuneConfig, n_event: int):
    import torch
    from torch import nn
    from transformers import AutoModel

    class MultiTaskEncoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = AutoModel.from_pretrained(cfg.base_model)
            hidden = self.encoder.config.hidden_size
            self.dropout = nn.Dropout(0.1)
            self.sentiment = nn.Linear(hidden, len(SENTIMENT_LABELS))
            self.event = nn.Linear(hidden, n_event)
            self.target = nn.Linear(hidden, len(SENTIMENT_LABELS))

        def pooled(self, input_ids, attention_mask, token_type_ids):
            hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids).last_hidden_state
            mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
            mean = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            cls = hidden[:, 0]
            # One pooling carries meaning (kept close to the teacher), the other carries the task features.
            return (cls, mean) if cfg.semantic_pooling == "cls" else (mean, cls)

        def forward(self, input_ids, attention_mask, token_type_ids):
            semantic, features = self.pooled(input_ids, attention_mask, token_type_ids)
            embedding = torch.nn.functional.normalize(semantic, dim=-1)
            features = self.dropout(features)
            return embedding, self.sentiment(features), self.event(features), self.target(features)

    return MultiTaskEncoder()


# --------------------------------------------------------------------------- training
def finetune(cfg: FinetuneConfig | None = None) -> dict:
    import torch
    from sklearn.metrics import accuracy_score, f1_score
    from torch.nn import functional as F
    from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

    cfg = cfg or FinetuneConfig()
    random.seed(cfg.seed), np.random.seed(cfg.seed), torch.manual_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""), flush=True)

    taxonomy = load_taxonomy()
    event_labels = tuple(taxonomy.ids)
    event_id = {name: i for i, name in enumerate(event_labels)}

    sent = _dev_split(sentiment_corpus())
    events = _dev_split(event_corpus(taxonomy))
    events["label"] = events["label"].map(event_id)
    targets = _dev_split(target_corpus())
    print(f"rows  sentiment {len(sent):,} | event {len(events):,} | target pairs {len(targets):,}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model)
    model = build_model(cfg, len(event_labels)).to(device)

    def encode(texts, pairs=None, max_length=cfg.max_length):
        enc = tokenizer(texts, pairs, truncation=True, max_length=max_length, padding=True, return_tensors="pt")
        if "token_type_ids" not in enc:
            enc["token_type_ids"] = torch.zeros_like(enc["input_ids"])
        return {k: v.to(device) for k, v in enc.items()}

    # Teacher embeddings: what the untouched base model thinks each text means.
    teacher = AutoModel.from_pretrained(cfg.base_model).to(device).eval()

    @torch.no_grad()
    def teacher_embed(texts: list[str]) -> torch.Tensor:
        enc = encode(texts)
        hidden = teacher(**enc).last_hidden_state
        if cfg.semantic_pooling == "cls":
            pooled = hidden[:, 0]
        else:
            mask = enc["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        return F.normalize(pooled, dim=-1)

    tasks = {
        "sentiment": sent[sent["split"] == "train"].reset_index(drop=True),
        "event": events[events["split"] == "train"].reset_index(drop=True),
        "target": targets[targets["split"] == "train"].reset_index(drop=True),
    }
    cw = {
        "sentiment": torch.tensor(class_weights(tasks["sentiment"]["label"].to_numpy(), 3), dtype=torch.float, device=device),
        "event": torch.tensor(class_weights(tasks["event"]["label"].to_numpy(), len(event_labels)), dtype=torch.float, device=device),
        "target": torch.tensor(class_weights(tasks["target"]["label"].to_numpy(), 3), dtype=torch.float, device=device),
    }
    steps_per_epoch = {name: math.ceil(len(df) / cfg.batch_size) for name, df in tasks.items()}
    total_steps = cfg.epochs * sum(steps_per_epoch.values())
    no_decay = ("bias", "LayerNorm.weight")
    optimizer = torch.optim.AdamW(
        [{"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": cfg.weight_decay},
         {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0}], lr=cfg.lr)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(cfg.warmup_fraction * total_steps), total_steps)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")

    @torch.no_grad()
    def predict(df: pd.DataFrame, head: str) -> np.ndarray:
        model.eval()
        out = []
        for start in range(0, len(df), 256):
            chunk = df.iloc[start:start + 256]
            if head == "target":
                enc = encode(chunk["entity"].tolist(), chunk["text"].tolist(), cfg.pair_max_length)
            else:
                enc = encode(chunk["text"].tolist())
            _, s, e, t = model(**enc)
            out.append({"sentiment": s, "event": e, "target": t}[head].float().softmax(-1).cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, 3))

    def evaluate(split: str) -> dict:
        report: dict = {"sentiment": {}, "target": {}}
        part = sent[sent["split"] == split]
        for name, grp in part.groupby("dataset"):
            proba = predict(grp, "sentiment")
            y = grp["label"].to_numpy()
            if name == "stocktwits":
                report["sentiment"][name] = {"n": len(grp), "directional_accuracy": float((((proba[:, 2] - proba[:, 0]) > 0) == (y == 2)).mean())}
            else:
                report["sentiment"][name] = {"n": len(grp), "accuracy": float(accuracy_score(y, proba.argmax(1))),
                                             "macro_f1": float(f1_score(y, proba.argmax(1), average="macro"))}
        ev = events[events["split"] == split]
        if split == "test":
            ev = ev[ev["origin"] == "human"]
        if len(ev):
            pred = predict(ev, "event").argmax(1)
            y = ev["label"].to_numpy()
            report["event"] = {"n": len(ev), "accuracy": float(accuracy_score(y, pred)),
                               "macro_f1": float(f1_score(y, pred, labels=sorted(set(y)), average="macro", zero_division=0))}
        tg = targets[targets["split"] == split]
        if len(tg):
            proba = predict(tg, "target")
            y = tg["label"].to_numpy()
            multi = (tg["n_entities"] > 1).to_numpy()
            report["target"]["all"] = {"n": len(tg), "accuracy": float(accuracy_score(y, proba.argmax(1)))}
            if multi.any():
                report["target"]["multi_entity"] = {"n": int(multi.sum()), "accuracy": float(accuracy_score(y[multi], proba.argmax(1)[multi]))}
        return report

    def dev_score(report: dict) -> float:
        parts = [m.get("macro_f1", m.get("directional_accuracy", 0.0)) for m in report["sentiment"].values()]
        parts.append(report.get("event", {}).get("macro_f1", 0.0))
        parts.append(report["target"].get("all", {}).get("accuracy", 0.0))
        return float(np.mean(parts))

    best_score, best_state, history = -1.0, None, []
    t_start = time.time()
    for epoch in range(cfg.epochs):
        model.train()
        order = [name for name, n in steps_per_epoch.items() for _ in range(n)]
        random.shuffle(order)
        perms = {name: np.random.permutation(len(df)) for name, df in tasks.items()}
        cursor = {name: 0 for name in tasks}
        running, seen = 0.0, 0
        for step, name in enumerate(order):
            idx = perms[name][cursor[name]: cursor[name] + cfg.batch_size]
            cursor[name] += cfg.batch_size
            batch = tasks[name].iloc[idx]
            labels = torch.tensor(batch["label"].to_numpy(), dtype=torch.long, device=device)
            texts = batch["text"].tolist()
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                if name == "target":
                    enc = encode(batch["entity"].tolist(), texts, cfg.pair_max_length)
                    _, _, _, logits = model(**enc)
                    loss = F.cross_entropy(logits.float(), labels, weight=cw[name])
                else:
                    enc = encode(texts)
                    embedding, s_logits, e_logits, _ = model(**enc)
                    logits = s_logits if name == "sentiment" else e_logits
                    per_row = F.cross_entropy(logits.float(), labels, weight=cw[name], reduction="none")
                    row_w = torch.tensor(batch["weight"].to_numpy(), dtype=torch.float, device=device)
                    loss = (per_row * row_w).sum() / row_w.sum()
                    distill = (1.0 - (embedding.float() * teacher_embed(texts)).sum(-1)).mean()
                    loss = loss + cfg.distill_weight * distill
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            running += float(loss)
            seen += 1
            if (step + 1) % 400 == 0:
                print(f"  epoch {epoch + 1} step {step + 1}/{len(order)} loss {running / seen:.4f} [{time.time() - t_start:.0f}s]", flush=True)
        report = evaluate("dev")
        score = dev_score(report)
        history.append({"epoch": epoch + 1, "train_loss": running / max(seen, 1), "dev_score": score})
        print(f"epoch {epoch + 1}: loss {running / max(seen, 1):.4f} | dev score {score:.4f} | "
              + " ".join(f"{k}={v.get('macro_f1', v.get('directional_accuracy', 0)):.3f}" for k, v in report["sentiment"].items())
              + f" event={report.get('event', {}).get('macro_f1', 0):.3f} target={report['target'].get('all', {}).get('accuracy', 0):.3f}",
              flush=True)
        if score > best_score:
            best_score = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    test_report = evaluate("test")

    # How well did clustering semantics survive? Compare pairwise similarities with the teacher's.
    sample = events[events["split"] == "test"]["text"].drop_duplicates().sample(400, random_state=0).tolist()
    with torch.no_grad():
        model.eval()
        student = torch.cat([model(**encode(sample[i:i + 100]))[0].float() for i in range(0, len(sample), 100)])
        teach = torch.cat([teacher_embed(sample[i:i + 100]).float() for i in range(0, len(sample), 100)])
    iu = np.triu_indices(len(sample), k=1)
    sim_s, sim_t = (student @ student.T).cpu().numpy()[iu], (teach @ teach.T).cpu().numpy()[iu]
    rank_corr = float(pd.Series(sim_s).corr(pd.Series(sim_t), method="spearman"))
    print(f"similarity preservation (Spearman vs original encoder): {rank_corr:.3f}", flush=True)

    # ------------------------------------------------------------------ export
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model = model.float().cpu().eval()
    dummy = tokenizer(["export sample"], return_tensors="pt")
    token_types = dummy.get("token_type_ids", torch.zeros_like(dummy["input_ids"]))
    fp32_path = MODEL_DIR / "model.fp32.onnx"
    export_kwargs = dict(
        input_names=["input_ids", "attention_mask", "token_type_ids"],
        output_names=["embedding", "sentiment", "event", "target"],
        dynamic_axes={n: {0: "batch", 1: "sequence"} for n in ("input_ids", "attention_mask", "token_type_ids")}
        | {n: {0: "batch"} for n in ("embedding", "sentiment", "event", "target")},
        opset_version=17,
    )
    export_args = (dummy["input_ids"], dummy["attention_mask"], token_types)
    try:  # newer PyTorch defaults to the dynamo exporter; the classic one is what ONNX Runtime quantises reliably
        torch.onnx.export(model, export_args, str(fp32_path), dynamo=False, **export_kwargs)
    except TypeError:
        torch.onnx.export(model, export_args, str(fp32_path), **export_kwargs)
    from onnxruntime.quantization import QuantType, quantize_dynamic

    int8_path = MODEL_DIR / "model.onnx"
    quantize_dynamic(str(fp32_path), str(int8_path), weight_type=QuantType.QUInt8)
    tokenizer.backend_tokenizer.save(str(MODEL_DIR / "tokenizer.json"))

    meta = {
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config": cfg.__dict__,
        "sentiment_labels": list(SENTIMENT_LABELS),
        "event_labels": list(event_labels),
        "train_rows": {name: len(df) for name, df in tasks.items()},
        "history": history,
        "test": test_report,
        "similarity_preservation_spearman": rank_corr,
        "files": {"model.onnx": round(int8_path.stat().st_size / 1e6, 1), "model.fp32.onnx": round(fp32_path.stat().st_size / 1e6, 1)},
        "minutes": round((time.time() - t_start) / 60, 1),
    }
    (MODEL_DIR / "model.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    out = DOCS_DIR / "results" / "finetune.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(test_report, indent=2))
    print(f"exported {int8_path} ({meta['files']['model.onnx']} MB int8; fp32 {meta['files']['model.fp32.onnx']} MB)")
    return meta


if __name__ == "__main__":
    finetune()
