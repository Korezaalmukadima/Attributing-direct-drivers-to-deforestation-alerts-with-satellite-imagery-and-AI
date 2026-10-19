"""
Generator data sintetis patch satelit 6-band untuk klasifikasi driver deforestasi.

Meniru desain input pada materi Dr. Bart Slachter:
  - 2 band Sentinel-1 (VV, VH)   -> radar, tembus awan
  - 4 band Sentinel-2 (B2,B3,B4,B8) -> optik, terganggu awan
  - 7 kelas driver, dengan distribusi sangat tidak seimbang (agriculture dominan)

PENTING: data ini SINTETIS untuk demonstrasi pipeline. Nilai pikselnya bukan
hasil pengukuran satelit sungguhan. Untuk skripsi, ganti dengan data asli
(lihat gee/export_patches.js).
"""
import numpy as np
from scipy.ndimage import gaussian_filter

CLASSES = [
    "smallholder_agri",     # 0 pertanian skala kecil
    "large_agri",           # 1 pertanian skala besar
    "road",                 # 2 pembangunan jalan
    "selective_logging",    # 3 penebangan selektif
    "mining",               # 4 tambang
    "wildfire",             # 5 kebakaran
    "natural_disturbance",  # 6 banjir / angin kencang
]
BAND_NAMES = ["S1_VV", "S1_VH", "S2_B2", "S2_B3", "S2_B4", "S2_B8"]
S1_IDX = [0, 1]
S2_IDX = [2, 3, 4, 5]

# Prior kelas: agriculture dominan (meniru ketidakseimbangan di materi)
PRIORS = np.array([0.40, 0.15, 0.08, 0.10, 0.04, 0.12, 0.11])

# Nilai rata-rata hutan utuh (sudah dinormalisasi kasar ke ~[0,1])
FOREST = np.array([0.55, 0.45, 0.05, 0.08, 0.05, 0.35], dtype=np.float32)


def _blob(rng, size, frac, smooth=3.0):
    """Mask tidak beraturan dengan luas ~frac dari patch."""
    noise = gaussian_filter(rng.standard_normal((size, size)), smooth)
    thr = np.quantile(noise, 1 - frac)
    return noise >= thr


def _rect(rng, size, h, w):
    m = np.zeros((size, size), bool)
    y = rng.integers(0, max(1, size - h))
    x = rng.integers(0, max(1, size - w))
    m[y:y + h, x:x + w] = True
    return m


def _line(rng, size, width=2):
    m = np.zeros((size, size), bool)
    y0, y1 = rng.integers(0, size, 2)
    for t in np.linspace(0, 1, size * 2):
        x = int(t * (size - 1))
        y = int(y0 + t * (y1 - y0))
        m[max(0, y - width // 2): y + width // 2 + 1, x] = True
    return m


def _apply(img, mask, delta):
    img[:, mask] += np.asarray(delta, dtype=np.float32)[:, None]


def make_patch(cls, rng, size=32):
    img = np.tile(FOREST[:, None, None], (1, size, size)).copy()

    if cls == 0:    # smallholder: bercak kecil, beberapa
        mask = np.zeros((size, size), bool)
        for _ in range(rng.integers(1, 4)):
            mask |= _rect(rng, size, rng.integers(3, 7), rng.integers(3, 7))
        _apply(img, mask, [-0.10, -0.08, 0.04, 0.05, 0.10, -0.12])
    elif cls == 1:  # large agri: area luas, bentuk kotak
        mask = _rect(rng, size, rng.integers(14, 24), rng.integers(14, 24))
        _apply(img, mask, [-0.12, -0.10, 0.05, 0.07, 0.12, -0.14])
    elif cls == 2:  # jalan: garis tipis tanah terbuka
        mask = _line(rng, size, width=2)
        _apply(img, mask, [-0.05, -0.06, 0.08, 0.09, 0.12, -0.18])
    elif cls == 3:  # selective logging: jalan + celah-celah kecil
        mask = _line(rng, size, width=1)
        _apply(img, mask, [-0.04, -0.05, 0.06, 0.07, 0.09, -0.12])
        gaps = np.zeros((size, size), bool)
        for _ in range(rng.integers(4, 9)):
            gaps |= _rect(rng, size, 2, 2)
        _apply(img, gaps, [-0.06, -0.05, 0.03, 0.04, 0.05, -0.08])
    elif cls == 4:  # tambang: lahan terbuka sangat terang, backscatter tinggi
        mask = _blob(rng, size, rng.uniform(0.15, 0.35), smooth=2.5)
        _apply(img, mask, [0.10, 0.05, 0.14, 0.15, 0.17, -0.20])
    elif cls == 5:  # wildfire: bekas bakar gelap di NIR, area tidak beraturan
        mask = _blob(rng, size, rng.uniform(0.25, 0.55), smooth=4.0)
        _apply(img, mask, [-0.04, -0.03, 0.01, 0.02, 0.03, -0.25])
    else:           # natural: banjir sepanjang sungai (radar sangat gelap)
        river = _line(rng, size, width=rng.integers(5, 10))
        _apply(img, river, [-0.35, -0.30, 0.01, 0.03, 0.01, -0.22])

    img += rng.normal(0, 0.02, img.shape).astype(np.float32)
    return np.clip(img, 0, 1)


def add_clouds(x, rng, prob=0.5):
    """Tutup band Sentinel-2 dengan awan pada sebagian sampel. x: (N,6,H,W)."""
    x = x.copy()
    n, _, h, w = x.shape
    for i in range(n):
        if rng.random() < prob:
            frac = rng.uniform(0.5, 1.0)
            cloud = _blob(rng, h, frac, smooth=5.0)
            x[i][np.ix_(S2_IDX, np.where(cloud.any(0))[0])] = x[i][np.ix_(S2_IDX, np.where(cloud.any(0))[0])]
            for b in S2_IDX:
                x[i, b][cloud] = 0.8 + rng.normal(0, 0.03, cloud.sum())
    return np.clip(x, 0, 1)


def make_dataset(n, seed=0, size=32):
    rng = np.random.default_rng(seed)
    y = rng.choice(len(CLASSES), size=n, p=PRIORS)
    x = np.stack([make_patch(c, rng, size) for c in y]).astype(np.float32)
    return x, y.astype(np.int64)
