"""Latih CNN driver deforestasi vs baseline RF; evaluasi OA & macro-F1; uji robustness awan.

Jalankan:  python src/train.py --epochs 15
"""
import argparse, json, os
import numpy as np, torch, torch.nn as nn
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

from synthetic_data import make_dataset, simulate_cloud, CLASSES
from model import DriverCNN
from baseline_rf import fit_rf, predict_rf


def predict(model, X, dev, bs=256):
    model.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            out.append(model(torch.from_numpy(X[i:i + bs]).to(dev)).argmax(1).cpu().numpy())
    return np.concatenate(out)


def report(name, y, p):
    oa, mf1 = accuracy_score(y, p), f1_score(y, p, average="macro")
    print(f"{name:<28} OA={oa:.3f}  macro-F1={mf1:.3f}")
    return {"OA": oa, "macroF1": mf1}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--n", type=int, default=300, help="sampel dasar per kelas")
    ap.add_argument("--out", default="outputs")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(0); np.random.seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    X, y = make_dataset(a.n)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=0)
    mu = Xtr.mean((0, 2, 3), keepdims=True); sd = Xtr.std((0, 2, 3), keepdims=True) + 1e-6
    # normalisasi per kanal (kanal S2 yang dinolkan ikut dinormalisasi -> model belajar sinyalnya)
    norm = lambda Z: ((Z - mu) / sd).astype(np.float32)
    print("Distribusi kelas:", dict(zip(CLASSES, np.bincount(y))))

    # ---- baseline RF
    rf = fit_rf(Xtr, ytr)
    res = {"RF (clear)": report("RF (clear)", yte, predict_rf(rf, Xte))}
    Xte_cloud = simulate_cloud(Xte, p=1.0)
    res["RF (S2 berawan)"] = report("RF (S2 berawan)", yte, predict_rf(rf, Xte_cloud))

    # ---- CNN (augmentasi: awan acak 30% + flip) dengan class weight
    model = DriverCNN().to(dev)
    w = torch.tensor(len(ytr) / (len(CLASSES) * np.bincount(ytr)), dtype=torch.float32).to(dev)
    lossf = nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.AdamW(model.parameters(), 2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 5e-3, total_steps=a.epochs * (len(ytr) // 64 + 1))
    for ep in range(a.epochs):
        model.train(); perm = np.random.permutation(len(ytr)); tot = 0
        for i in range(0, len(perm), 64):
            idx = perm[i:i + 64]
            xb = simulate_cloud(Xtr[idx], 0.3, seed=np.random.randint(1e9))
            if np.random.rand() < .5: xb = xb[:, :, :, ::-1]
            xb = torch.from_numpy(np.ascontiguousarray(norm(xb))).to(dev)
            yb = torch.from_numpy(ytr[idx]).to(dev)
            opt.zero_grad(); l = lossf(model(xb), yb); l.backward(); opt.step(); sched.step()
            tot += l.item() * len(idx)
        print(f"epoch {ep + 1:02d}  loss={tot / len(ytr):.4f}")

    p_clear = predict(model, norm(Xte), dev)
    res["CNN (clear)"] = report("CNN (clear)", yte, p_clear)
    res["CNN (S2 berawan)"] = report("CNN (S2 berawan)", yte, predict(model, norm(Xte_cloud), dev))
    res["CNN (S1 saja, S2=0)"] = res["CNN (S2 berawan)"]

    print("\n", classification_report(yte, p_clear, target_names=CLASSES, digits=3))
    json.dump(res, open(f"{a.out}/metrics.json", "w"), indent=2)

    cm = confusion_matrix(yte, p_clear, normalize="true")
    fig, ax = plt.subplots(figsize=(7, 6)); im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(7)); ax.set_yticks(range(7))
    ax.set_xticklabels(CLASSES, rotation=45, ha="right"); ax.set_yticklabels(CLASSES)
    for i in range(7):
        for j in range(7): ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", fontsize=7)
    ax.set_title("Confusion matrix (CNN, ternormalisasi)"); fig.colorbar(im); fig.tight_layout()
    fig.savefig(f"{a.out}/confusion_matrix.png", dpi=150)
    torch.save(model.state_dict(), f"{a.out}/driver_cnn.pt")


if __name__ == "__main__":
    main()
