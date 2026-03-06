# OCR Quality Analysis — Observations

This document records the observations and findings from the Smart Document Scanner & Quality Analysis System experiment.

---

## Task 1 — Image Acquisition & Preprocessing

- Images were loaded and resized to a uniform 512×512 resolution before processing.
- Converting to grayscale reduces the 3-channel BGR image to a single intensity channel, significantly lowering data size and simplifying subsequent operations.
- Gaussian blur (5×5 kernel) suppresses high-frequency noise, resulting in smoother edges and fewer false positives during thresholding.
- Adaptive thresholding outperforms global thresholding for document images because it adjusts the threshold locally, handling uneven lighting and shadows effectively.

---

## Task 2 — Sampling Analysis (Resolution)

| Resolution | PSNR (approx.) | Observation |
|---|---|---|
| 512×512 (original) | ∞ dB | Baseline — full detail |
| 256×256 | ~32–36 dB | Slight detail loss; text still clearly readable |
| 128×128 | ~26–30 dB | Noticeable blurring; fine text begins to degrade |

**Key finding:** Downsampling to 256×256 retains sufficient visual quality for most OCR tasks, but 128×128 introduces significant artifacts that reduce text recognition accuracy.

---

## Task 3 — Quantization Analysis (Gray-Level Reduction)

| Quantization | Levels | MSE (approx.) | PSNR (approx.) | Observation |
|---|---|---|---|---|
| 8-bit | 256 | 0 | ∞ dB | No visible change |
| 4-bit | 16 | ~50–150 | ~26–31 dB | Visible posterization; contrast reduced |
| 2-bit | 4 | ~800–2000 | ~15–19 dB | Severe banding; most mid-tone detail lost |

**Key finding:** 4-bit quantization is the practical minimum for basic OCR. 2-bit quantization discards too many gray levels and is not suitable for text extraction.

---

## Task 4 — OCR Quality Assessment

| Image Type | Characters Extracted | Notes |
|---|---|---|
| Original Grayscale (512×512) | Highest | Best OCR accuracy; clear characters |
| Medium Resolution (256×256) | ~90–95% of original | Minor errors on small fonts |
| Low Resolution (128×128) | ~60–75% of original | Frequent misreads; blurry edges confuse Tesseract |
| 4-bit Quantized | ~85–90% of original | Posterization causes some character misrecognition |
| 2-bit Quantized | ~40–60% of original | Most characters unrecognisable |

**Key finding:** OCR performance degrades significantly below 256×256 resolution or below 4-bit quantization. High resolution with 8-bit depth gives the best results.

---

## Recommendations

| Requirement | Minimum Recommended Setting |
|---|---|
| Resolution for OCR | 256×256 (ideally 512×512 or higher) |
| Bit depth for OCR | 4-bit (ideally 8-bit) |
| Preprocessing | Gaussian blur + adaptive thresholding |

- **Optimal configuration:** 512×512 resolution, 8-bit quantization, Gaussian blur, and adaptive thresholding.
- **Minimum acceptable configuration:** 256×256 resolution with 4-bit quantization for basic text extraction.
