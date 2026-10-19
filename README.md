# Klasifikasi Driver Deforestasi Tropis dengan CNN Multi-Sensor (Sentinel-1 + Sentinel-2)

**resume materi kuliah tamu**, **insight untuk skripsi**, dan **studi kasus + implementasi kode**
berdasarkan presentasi *Dr. Bart Slachter* (Wageningen University) tentang deforestation alerts dan atribusi *direct driver*
berbasis AI.

---

## 1. Resume Materi

### 1.1 Latar belakang
- Hutan tropis penting untuk penyimpanan karbon, regulasi siklus air, keanekaragaman hayati, dan mata pencaharian.
- Hutan hilang cepat akibat perubahan penggunaan lahan, penebangan kayu, dan pertambangan.

### 1.2 Data satelit
| Sensor | Kelebihan | Kekurangan |
|---|---|---|
| **Sentinel-2 (optik)** | Informasi kesehatan & perubahan vegetasi, mudah diinterpretasi | Sering tertutup awan di wilayah tropis |
| **Sentinel-1 (radar C-band)** | Menembus awan, data konsisten, sensitif air/kelembapan | Lebih sulit diinterpretasi, penetrasi kanopi rendah |

Keduanya digabung sebagai *multi-band image stack* (S1: 2 band; S2: 4 band).

### 1.3 Deforestation alerts
- Citra dianalisis ±mingguan pada resolusi 10-30 m untuk mendeteksi gangguan hutan (contoh: **RADD Alerts**, berbasis Sentinel-1; **GLAD Alerts**).
- Keterbatasan: alert hanya menunjukkan *ada gangguan*, bukan *penyebabnya*.

### 1.4 Klasifikasi driver dengan CNN
- Input: stack S1+S2 setelah alert terdeteksi (dibuat tiap bulan).
- Output 7 kelas: *smallholder agriculture, large-scale agriculture, road development, selective logging, mining, wildfire, natural disturbance* (banjir, windthrow) + label *unconfident*.
- Model membaca **ukuran, bentuk, dan konteks sekitar** alert. Kelas pertanian dipecah lagi berdasarkan **ukuran** dan **keberadaan api**.
- Model **belajar sendiri** mengandalkan radar saat optik tertutup awan (tidak ada aturan eksplisit).

### 1.4 Temuan regional (dataset 3 tahun)
- **Amerika Selatan (Brasil, Bolivia):** dominasi pertanian skala besar & kebakaran; kekeringan ekstrem 2024 → degradasi, belum tentu deforestasi permanen.
- **Afrika Tengah:** pertanian skala kecil dominan; selective logging luas di Gabon & Kongo.
- **Asia Tenggara (Indonesia, Malaysia):** campuran pertanian besar/kecil, logging, dan tambang skala besar.
- **Pola sekuens:** tambang emas DRC 2022 → pertanian meningkat di sekitarnya; Bolivia: jalan + logging (2022-2023) → kebakaran di area yang sama.
- **Musiman:** wildfire & pertanian paling musiman; Asia Tenggara kurang musiman karena musim kering tidak setegas Amazon/Kongo.

### 1.5 Akurasi
- *Overall accuracy* tinggi, tetapi **didominasi kelas pertanian**.
- **Macro-F1** lebih rendah karena semua 7 kelas berbobot sama → penilaian lebih jujur.
- Kelas tersulit: **wildfire & natural disturbance (banjir)** — sering tertukar (kebakaran sering terjadi di dataran banjir). Mitigasi: *retraining* dengan lebih banyak contoh berlabel.

### 1.6 Keterbatasan
1. Konsistensi cakupan satelit (satelit rusak → deteksi hilang).
2. Hanya *direct driver* saat deteksi; tujuan akhir lahan baru terlihat belakangan (mis. bekas api → kebun).
3. Update bulanan → latensi; tidak cocok untuk intervensi *real-time*.
4. Aktivitas seperti *artisanal logging* dan produksi arang tidak bisa dibedakan dari citra saja → perlu data lapangan.

### 1.7 Akses data & pemanfaatan
- Dipublikasikan bulanan di **Global Nature Watch** (sebelumnya Global Forest Watch), dapat di-overlay dengan konsesi, kawasan lindung, tenurial lahan.
- Pemantauan kebakaran real-time lebih tepat memakai sensor termal (harian/mingguan, mis. NASA FIRMS).
- Penggunaan oleh penegak hukum: alert dasar (GLAD/RADD) sudah dipakai di lapangan; lapisan driver masih baru (hitungan bulan).

### 1.8 Poin Q&A
- **Deep learning vs ML konvensional:** mulai dari baseline konvensional; gunakan DL jika masalah kompleks (konteks/time series), data berlabel banyak, dan biaya komputasi sepadan.
- **S1 vs S2 untuk banjir:** S1 lebih sensitif air, namun C-band terbatas di bawah kanopi rapat.
- **Satelit saja cukup?** Tidak; data lapangan penting untuk driver yang lebih bernuansa.
- **Kolaborasi:** kunci perbaikan model = *ground truth* berkualitas.

---

## 2. Insight untuk Skripsi

### 2.1 Gap & peluang yang relevan untuk Indonesia
1. **Driver deforestasi Sumatra Selatan/Bengkulu/Kalimantan**: sawit, HTI, tambang batubara, karhutla lahan gambut. Model global belum tentu optimal untuk konteks lokal.
2. **Pembedaan karhutla vs banjir/natural** (kelas tersulit) → kontribusi: menambah fitur (indeks NBR/dNBR, hotspot FIRMS, jarak ke sungai, DEM/HAND, tipe tanah gambut).
3. **Radar-optik fusion saat berawan** di Indonesia → uji robustness terhadap persentase tutupan awan.
4. **Integrasi data FIRMS** (sudah pernah Anda kerjakan) sebagai fitur tambahan untuk memisahkan *fire-driven* vs *non-fire* clearing.
5. **Analisis sekuens/ko-okurens** (jalan → logging → api; tambang → pertanian) berbasis *time series* alert.

### 2.2 Contoh judul skripsi
1. *Klasifikasi Driver Deforestasi di Sumatera Selatan Menggunakan CNN dengan Fusi Sentinel-1 dan Sentinel-2.*
2. *Pembedaan Kebakaran Hutan dan Banjir pada Alert Deforestasi Menggunakan Fitur Termal FIRMS, Radar, dan Optik.*
3. *Perbandingan Random Forest dan CNN untuk Atribusi Penyebab Deforestasi pada Kondisi Tutupan Awan Tinggi.*
4. *Analisis Sekuens Aktivitas Pembukaan Lahan (Jalan-Logging-Kebakaran) pada Lahan Konsesi di Indonesia.*

### 2.3 Rumusan masalah & desain eksperimen
- **RM1:** Seberapa akurat CNN mengklasifikasikan driver dibanding Random Forest (baseline)?
- **RM2:** Seberapa besar kontribusi S1 vs S2 vs gabungan (ablation) pada berbagai tingkat tutupan awan?
- **RM3:** Apakah fitur tambahan (NBR, FIRMS, DEM) memperbaiki F1 kelas wildfire & natural?
- Metrik: OA, **macro-F1**, F1 per kelas, confusion matrix, uji pada split **spasial** (hindari kebocoran spasial).
- Ablation: {S1}, {S2}, {S1+S2}, {S1+S2+fitur tambahan}; CNN vs RF vs (opsional) U-Net/ResNet.

### 2.4 Sumber data
- Alert: RADD (`projects/radar-wur/raddalert/v1`), GLAD/GFW Integrated Alerts.
- Citra: `COPERNICUS/S1_GRD`, `COPERNICUS/S2_SR_HARMONIZED`.
- Hotspot: NASA FIRMS. Batas konsesi/kawasan: Global Nature Watch, KLHK/Geoportal.
- Label: interpretasi visual (Google Earth/Planet NICFI), data lapangan, atau dataset driver publik (mis. peta driver WRI/WUR).

### 2.5 Tips metodologis
- Gunakan **class weight / focal loss** dan laporkan **macro-F1**, bukan hanya OA.
- Hindari *data leakage*: pisahkan train/test per wilayah, bukan acak per piksel.
- Augmentasi **simulasi awan** agar model belajar bergantung pada radar.
- Jelaskan keterbatasan *direct driver* & label *unconfident*.

---

## 3. Studi Kasus & Implementasi

**Kasus:** membangun pipeline klasifikasi 7 kelas driver pada patch 32x32 px (10 m) dari 6 kanal
(S1: VV, VH; S2: B2, B3, B4, B8), meniru pendekatan materi, dan menguji **robustness saat optik tertutup awan**.

### Struktur
```
src/
  synthetic_data.py   # generator data sintetis 7 kelas + simulasi awan (reproducible tanpa GEE)
  model.py            # CNN 3 blok konvolusi
  baseline_rf.py      # benchmark Random Forest
  train.py            # latih, evaluasi OA/macro-F1, uji awan, confusion matrix
  gee_export.py       # ekspor patch nyata dari Google Earth Engine
  load_tifs.py        # GeoTIFF -> dataset.npz
```

### Menjalankan
```bash
pip install -r requirements.txt
python src/train.py --epochs 15      # output: outputs/metrics.json, confusion_matrix.png
```

### Hasil uji baseline RF pada data sintetis
| Kondisi | OA | Macro-F1 |
|---|---|---|
| Citra jernih (S1+S2) | 0.979 | 0.977 |
| S2 tertutup awan | 0.469 | 0.166 |

Baseline RF **runtuh saat optik hilang**, sehingga motivasi augmentasi awan + fusi radar menjadi jelas. Isi tabel hasil
CNN dengan keluaran `outputs/metrics.json` setelah Anda menjalankan `train.py`.

> **Catatan jujur:** data sintetis hanya untuk mendemonstrasikan pipeline; angkanya **bukan** hasil ilmiah.
> Untuk skripsi, gunakan data nyata (`gee_export.py` + `load_tifs.py`), lalu ubah `train.py` agar memuat `data/dataset.npz`
> (`d = np.load(...); X, y = d['X'], d['y']`). Kode CNN belum dijalankan di lingkungan pembuatan (PyTorch tidak terpasang),
> jadi cek dulu di mesin Anda. `gee_export.py` juga belum diuji.

### Pengembangan lanjutan
- Tambah kanal: NBR/dNBR, jumlah hotspot FIRMS, DEM/HAND, jarak ke jalan/sungai.
- Input *before/after* (12 kanal) atau time series (CNN-LSTM/Transformer).
- Kelas *unconfident* melalui ambang probabilitas softmax.
- Evaluasi spasial (*spatial block cross-validation*) dan Grad-CAM untuk interpretasi.

## Referensi
- Presentasi kuliah tamu Dr. Bart Slachter (Wageningen University).
- Reiche et al. (2021), RADD alerts; Hansen et al. (2016), GLAD alerts; Global Nature Watch / Global Forest Watch.
