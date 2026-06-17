"""Phase 6 fine-tune — runs ENTIRELY on this machine (no Colab, no upload).

Trains a small projection HEAD on top of FROZEN DINOv2 embeddings, using the natural
supervision already in the dataset: a rug's base image + its g/l/t variants are the
SAME rug, so they should embed nearby. The `t` variant is held out for an honest
before/after recall check. The trained head (~2 MB) is written next to the FAISS
indices and gets wired into search.py as a "tuned" encoder.

It reuses the project's OWN DINOv2 encoder and config paths, so there is nothing to
download again and no path to set — it reads the same data/catalogue the app uses.

    python -m scripts.finetune_local        # one-time; minutes on CPU

Why a frozen backbone + head (not a full fine-tune): ~1,136 images is tiny, so a frozen
backbone can't overfit; embedding is the only real cost and it happens once.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from carpet_search.config import load_settings
from carpet_search.embedding import get_dino

# Must match the architecture the integration step reconstructs (saved in the .pt).
D_IN, D_HIDDEN, D_OUT = 768, 512, 256


def collect(settings) -> list[tuple[str, str, str]]:
    """(path, sku, variant). base = catalogue image; g/l/t = lifestyle variants."""
    rows: list[tuple[str, str, str]] = []
    for p in sorted(settings.paths.catalogue_images.glob("*.jpg")):
        rows.append((str(p), p.stem, "base"))
    for p in sorted(settings.paths.lifestyle_images.glob("*.jpg")):
        sku, _, var = p.stem.partition("__")
        rows.append((str(p), sku, var or "base"))
    return rows


class Head(nn.Module):
    def __init__(self, d_in=D_IN, d_hidden=D_HIDDEN, d_out=D_OUT):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, d_hidden), nn.GELU(), nn.Linear(d_hidden, d_out))

    def forward(self, x):
        return F.normalize(self.net(x), dim=1)


def supcon_loss(z, y, temp=0.1):
    """Supervised contrastive loss: pull same-rug embeddings together, push others apart."""
    sim = (z @ z.T) / temp
    sim = sim - sim.max(dim=1, keepdim=True).values.detach()
    same = (y[:, None] == y[None, :]).float()
    eye = torch.eye(len(y), device=z.device)
    same = same - eye                                   # exclude self-pairs
    exp = torch.exp(sim) * (1 - eye)
    log_prob = sim - torch.log(exp.sum(1, keepdim=True) + 1e-9)
    pos = (same * log_prob).sum(1) / same.sum(1).clamp(min=1)
    return -pos[same.sum(1) > 0].mean()


def recall_at_k(qfeat, gfeat, qlab, glab, ks=(1, 5, 10)):
    sims = qfeat @ gfeat.T
    order = sims.argsort(dim=1, descending=True)
    rec = {k: 0 for k in ks}
    for i in range(qfeat.shape[0]):
        ranked = glab[order[i]]
        for k in ks:
            if (ranked[:k] == qlab[i]).any():
                rec[k] += 1
    n = max(1, qfeat.shape[0])
    return {f"R@{k}": round(rec[k] / n, 3) for k in ks}


def main() -> None:
    settings = load_settings()
    rows = collect(settings)
    if not rows:
        raise SystemExit(
            f"No .jpg images under {settings.paths.catalogue_images} / "
            f"{settings.paths.lifestyle_images}. Run ingest + build_index first."
        )
    skus = sorted({s for _, s, _ in rows})
    sku2id = {s: i for i, s in enumerate(skus)}
    print(f"{len(rows)} images, {len(skus)} rugs, "
          f"variants={dict(Counter(v for _, _, v in rows))}")

    # --- frozen DINOv2: embed every image once (reuses the app's cached encoder) ---
    dino = get_dino(settings.models.dino_name, settings.models.image_size,
                    settings.preprocess.grayscale)
    device = dino.device
    print(f"embedding {len(rows)} images with frozen DINOv2 on {device} "
          f"(the first batch is slow while the model warms up) ...")
    paths = [r[0] for r in rows]
    bs = 16
    chunks = []
    for i in tqdm(range(0, len(paths), bs), desc="DINO embed", unit="batch"):
        imgs = [Image.open(p).convert("RGB") for p in paths[i:i + bs]]
        chunks.append(dino.embed_images(imgs))
    X = torch.from_numpy(np.vstack(chunks).astype("float32"))   # (N, 768), L2-normalised
    labels = torch.tensor([sku2id[r[1]] for r in rows])
    variants = np.array([r[2] for r in rows])

    # --- split: hold out `t` (fallback l/g) as the eval query set ---
    held = next((v for v in ("t", "l", "g") if (variants == v).any()), None)
    if held is None:
        raise SystemExit(f"need g/l/t variants to train/eval; saw {set(variants)}")
    is_held = variants == held
    is_base = variants == "base"
    if not is_base.any():
        is_base = ~is_held
    train_idx = np.where(~is_held)[0]
    gallery_idx = np.where(is_base)[0]
    query_idx = np.where(is_held)[0]
    print(f"held-out='{held}'  train={len(train_idx)}  gallery={len(gallery_idx)}  "
          f"queries={len(query_idx)}")

    print("BEFORE (raw DINOv2):",
          recall_at_k(X[query_idx], X[gallery_idx], labels[query_idx], labels[gallery_idx]))

    # --- train the projection head (supervised contrastive) ---
    head = Head().to(device)
    opt = torch.optim.Adam(head.parameters(), lr=1e-3, weight_decay=1e-4)
    Xtr, ytr = X[train_idx].to(device), labels[train_idx].to(device)
    print("training projection head (300 epochs) ...")
    for epoch in range(300):
        perm = torch.randperm(len(Xtr), device=device)
        losses = []
        for i in range(0, len(perm), 256):
            b = perm[i:i + 256]
            if len(b) < 8:
                continue
            loss = supcon_loss(head(Xtr[b]), ytr[b])
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
        if (epoch + 1) % 50 == 0:
            print(f"  epoch {epoch + 1:3d}  loss {np.mean(losses):.4f}")

    # --- eval AFTER + save ---
    head.eval()
    with torch.no_grad():
        Xh = head(X.to(device)).cpu()
    print("AFTER  (DINOv2 + head):",
          recall_at_k(Xh[query_idx], Xh[gallery_idx], labels[query_idx], labels[gallery_idx]))

    out = settings.paths.index_dir / "dino_head_finetuned.pt"
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": head.state_dict(), "d_in": D_IN, "d_hidden": D_HIDDEN, "d_out": D_OUT},
               str(out))
    print(f"saved -> {out}\n"
          "Next: I wire this head into search.py as the tuned encoder and re-measure.")


if __name__ == "__main__":
    main()
