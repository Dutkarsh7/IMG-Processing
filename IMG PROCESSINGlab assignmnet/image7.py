import cv2
import numpy as np
import matplotlib.pyplot as plt

# ---------------- Read Image ----------------
img = cv2.imread('images.webp')

if img is None:
    print("Error: Image not found")
    exit()

rows, cols = img.shape[:2]

# ---------------- Translation ----------------
M1 = np.float32([[1, 0, 100],
                 [0, 1, 50]])
translated = cv2.warpAffine(img, M1, (cols, rows))

# ---------------- Scaling ----------------
scaled = cv2.resize(img, None, fx=1.5, fy=1.5)

# ---------------- Rotation (No Cropping) ----------------
angle = 45
(h, w) = img.shape[:2]
center = (w // 2, h // 2)

M2 = cv2.getRotationMatrix2D(center, angle, 1.0)

cos = np.abs(M2[0, 0])
sin = np.abs(M2[0, 1])

new_w = int((h * sin) + (w * cos))
new_h = int((h * cos) + (w * sin))

M2[0, 2] += (new_w / 2) - center[0]
M2[1, 2] += (new_h / 2) - center[1]

rotated = cv2.warpAffine(img, M2, (new_w, new_h))

# ---------------- Convert BGR to RGB ----------------
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
translated_rgb = cv2.cvtColor(translated, cv2.COLOR_BGR2RGB)
scaled_rgb = cv2.cvtColor(scaled, cv2.COLOR_BGR2RGB)
rotated_rgb = cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB)

# ---------------- Display ----------------
plt.figure(figsize=(10, 8))

plt.subplot(2, 2, 1)
plt.imshow(img_rgb)
plt.title("Original")
plt.axis("off")

plt.subplot(2, 2, 2)
plt.imshow(translated_rgb)
plt.title("Translated")
plt.axis("off")

plt.subplot(2, 2, 3)
plt.imshow(scaled_rgb)
plt.title("Scaled")
plt.axis("off")

plt.subplot(2, 2, 4)
plt.imshow(rotated_rgb)
plt.title("Rotated (No Cropping)")
plt.axis("off")

plt.tight_layout()
plt.show()