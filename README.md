# IMG-Processing

A collection of Image Processing lab assignments covering geometric transformations, document scanning, and OCR quality analysis using Python, OpenCV, and Tesseract.

---

## Repository Structure

```
IMG-Processing/
├── img-processing-lab/          # Lab: Geometric Image Transformations
│   ├── image7.py                # Translation, scaling, and rotation demo
│   └── images.webp              # Sample input image
│
├── ocr-lab-1/                   # Lab: Smart Document Scanner & OCR Analysis
│   ├── app.py                   # Flask web application for document scanning
│   ├── scanner.py               # CLI-based scanner & quality analysis tool
│   ├── requirements.txt         # Python dependencies
│   ├── OBSERVATIONS.md          # OCR quality observations and analysis notes
│   ├── templates/
│   │   └── index.html           # Web UI template
│   ├── outputs/                 # (generated) processed images & OCR results
│   └── uploads/                 # (generated) user-uploaded documents
│
└── README.md
```

---

## Labs

### 1. Geometric Image Transformations (`img-processing-lab/`)

Demonstrates fundamental geometric transformations on images using OpenCV:

| Transformation | Description |
|---|---|
| **Translation** | Shifts the image by a specified offset |
| **Scaling** | Resizes the image by a given factor |
| **Rotation** | Rotates the image without cropping (expands canvas) |

**Run:**

```bash
cd img-processing-lab
python image7.py
```

> **Requirements:** `opencv-python`, `numpy`, `matplotlib`

---

### 2. Smart Document Scanner & OCR Analysis (`ocr-lab-1/`)

A full document scanning pipeline that covers:

- **Image Acquisition & Preprocessing** — resize, grayscale, Gaussian blur, adaptive thresholding
- **Sampling Analysis** — compare image quality at 512×512, 256×256, and 128×128 resolutions using PSNR
- **Quantization Analysis** — evaluate 8-bit, 4-bit, and 2-bit gray-level quantization with MSE/PSNR metrics
- **OCR Quality Assessment** — extract text with Tesseract across varying image qualities

#### Option A — Web Interface (`app.py`)

```bash
cd ocr-lab-1
pip install -r requirements.txt
python app.py
```

Open [http://localhost:5000](http://localhost:5000) in your browser, upload a document image, and view the scan results.

#### Option B — Command-Line Tool (`scanner.py`)

```bash
cd ocr-lab-1
pip install -r requirements.txt
python scanner.py
```

Enter the path to your document image when prompted. Outputs are saved to the `outputs/` folder.

> **Requirements:** `opencv-python`, `numpy`, `matplotlib`, `pytesseract`, `Pillow`, `Flask`  
> **External dependency:** [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki) must be installed on your system.

---

## Prerequisites

- Python 3.8+
- [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki) (for OCR lab)

Install Python dependencies for a specific lab:

```bash
pip install -r ocr-lab-1/requirements.txt
```
