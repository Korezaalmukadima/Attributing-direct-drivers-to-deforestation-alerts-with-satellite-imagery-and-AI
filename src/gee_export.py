"""Ekspor patch Sentinel-1 + Sentinel-2 (10 m, 32x32 px) di sekitar titik alert ke Google Drive.

Persiapan:  pip install earthengine-api pandas ; earthengine authenticate
Input CSV (data/alerts_labeled.csv): kolom lon, lat, date (YYYY-MM-DD), label (0-6), id
Label diperoleh dari interpretasi visual (mis. Google Earth / Planet) terhadap alert RADD/GLAD.
Catatan: skrip ini belum diuji end-to-end; sesuaikan nama aset/band bila API berubah.
"""
import ee, pandas as pd

ee.Initialize()
PATCH_PX, SCALE = 32, 10
BANDS_S2 = ["B2", "B3", "B4", "B8"]


def build_stack(lon, lat, date, window_days=30):
    pt = ee.Geometry.Point([lon, lat])
    d0 = ee.Date(date)
    s1 = (ee.ImageCollection("COPERNICUS/S1_GRD")
          .filterBounds(pt).filterDate(d0, d0.advance(window_days, "day"))
          .filter(ee.Filter.eq("instrumentMode", "IW"))
          .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
          .select(["VV", "VH"]).median())
    s2 = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
          .filterBounds(pt).filterDate(d0, d0.advance(window_days, "day"))
          .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 70))
          .select(BANDS_S2).median().divide(10000))
    # Jika tidak ada citra S2 yang lolos, band S2 akan kosong -> ditangani saat loading (diisi 0)
    region = pt.buffer(PATCH_PX * SCALE / 2).bounds()
    return s1.addBands(s2), region


def export_all(csv="data/alerts_labeled.csv", folder="deforestation_patches"):
    df = pd.read_csv(csv)
    for r in df.itertuples():
        img, region = build_stack(r.lon, r.lat, r.date)
        ee.batch.Export.image.toDrive(
            image=img.toFloat(), description=f"patch_{r.id}_c{r.label}", folder=folder,
            region=region, scale=SCALE, crs="EPSG:4326", fileFormat="GeoTIFF").start()
    print(f"{len(df)} task diekspor ke Google Drive/{folder}")


if __name__ == "__main__":
    export_all()
