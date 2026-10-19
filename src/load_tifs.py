"""Ubah GeoTIFF hasil gee_export.py (nama: patch_<id>_c<label>.tif) menjadi data/dataset.npz."""
import glob, re, sys, numpy as np, rasterio


def main(folder="data/tifs", out="data/dataset.npz", size=32):
    X, y = [], []
    for f in sorted(glob.glob(f"{folder}/patch_*_c*.tif")):
        with rasterio.open(f) as s:
            a = np.nan_to_num(s.read().astype("float32"))[:, :size, :size]   # (6, H, W)
        if a.shape[1:] != (size, size):
            continue
        X.append(a); y.append(int(re.search(r"_c(\d+)\.tif$", f).group(1)))
    np.savez_compressed(out, X=np.stack(X), y=np.array(y))
    print("Tersimpan:", out, np.stack(X).shape)


if __name__ == "__main__":
    main(*sys.argv[1:])
