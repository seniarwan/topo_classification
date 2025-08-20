[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1039mDn3_DqvFpiCit0xY2yzqqB670eIU#offline=true&sandboxMode=true)

# Topographic Classification - Pure Python

Implementasi klasifikasi topografi Iwahashi & Pike (2007) menggunakan Python.

## Features

- ✅ **Pure Python** - Tidak memerlukan ArcGIS/ArcPy
- ✅ **Simple API** - Mudah digunakan
- ✅ **Google Colab Ready** - Siap digunakan di cloud
- ✅ **Modular** - Dapat dikustomisasi
- ✅ **Fast** - Optimized dengan NumPy/SciPy

## Quick Start

### Instalasi di Google Colab

```python
# Clone repository
!git clone https://github.com/seniarwan/topographic-classification.git
%cd topographic-classification

# Install
!pip install -e .

# Import dan gunakan
from topoclassify import quick_classify

# Klasifikasi cepat
result = quick_classify("path/to/dem.tif")
```

### Instalasi Lokal

```bash
git clone https://github.com/username/topographic-classification.git
cd topographic-classification
pip install -e .
```

## Usage

### Simple Usage

```python
from topoclassify import quick_classify

# Klasifikasi dengan 1 baris kode
result = quick_classify("dem.tif", output_path="result.tif")
```

### Advanced Usage

```python
from topoclassify import TopographicClassifier
from topoclassify.utils import plot_results, get_class_stats

# Inisialisasi
classifier = TopographicClassifier("dem.tif")

# Akses metrik individual
slope = classifier.slope
convexity = classifier.convexity
texture = classifier.texture

# Klasifikasi
classification = classifier.classify()

# Visualisasi
plot_results(classification, slope, convexity, texture)

# Statistik
stats = get_class_stats(classification, slope, convexity, texture)
```

## Input Requirements

- **Format**: GeoTIFF, atau format yang didukung GDAL
- **Proyeksi**: Metric coordinate system (untuk perhitungan slope yang akurat)
- **Resolusi**: Disarankan 30m atau lebih halus

## Output

Klasifikasi menghasilkan 24 kelas topografi berdasarkan kombinasi:
- **Slope** (tinggi/rendah)
- **Convexity** (cembung/cekung) 
- **Texture** (kasar/halus)
- **3 level hierarki**

## Dependencies

```
numpy>=1.20.0
scipy>=1.7.0
rasterio>=1.3.0
scikit-image>=0.19.0
matplotlib>=3.5.0
```

## Examples

Lihat folder `examples/` untuk tutorial lengkap:
- `basic_usage.ipynb` - Penggunaan dasar
- `colab_demo.ipynb` - Demo untuk Google Colab

## References

Iwahashi, J., & Pike, R. J. (2007). Automated classifications of topography from DEMs by an unsupervised nested-means algorithm and a three-part geometric signature. *Geomorphology*, 86(3-4), 409-440.
