"""
================================================================================
Smart Document Scanner & Quality Analysis System
================================================================================
Name: [Your Name]
Roll No: [Your Roll Number]
Course: Image Processing & Computer Vision
Unit: Mini Project
Assignment Title: Smart Document Scanner & Quality Analysis System
Date: February 24, 2026
================================================================================
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from datetime import datetime
import pytesseract

# Configure Tesseract path (Windows)
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


def print_welcome():
    """Display welcome message"""
    print("\n" + "="*80)
    print(" "*20 + "SMART DOCUMENT SCANNER & QUALITY ANALYSIS SYSTEM")
    print("="*80)
    print("\nCourse: Image Processing & Computer Vision")
    print("Project: Document Quality Analysis with OCR")
    print("Date:", datetime.now().strftime("%B %d, %Y"))
    print("\nThis system analyzes document image quality through:")
    print("  • Image Acquisition & Preprocessing")
    print("  • Resolution Sampling Analysis")
    print("  • Quantization Analysis")
    print("  • OCR Quality Assessment")
    print("="*80 + "\n")


def create_output_folder():
    """Create outputs folder if it doesn't exist"""
    if not os.path.exists('outputs'):
        os.makedirs('outputs')
        print("[✓] Created 'outputs' folder")


def load_and_preprocess_image(image_path):
    """
    Load image and perform preprocessing
    Returns: original, resized, grayscale, blurred, thresholded images
    """
    print("\n[TASK 1] Image Acquisition & Preprocessing")
    print("-" * 60)
    
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print(f"[✗] Error: Could not load image from {image_path}")
        sys.exit(1)
    
    print(f"[✓] Image loaded successfully: {image_path}")
    print(f"    Original size: {img.shape[1]}x{img.shape[0]}")
    
    # Resize to 512x512
    img_resized = cv2.resize(img, (512, 512))
    print(f"[✓] Resized to: 512x512")
    
    # Convert to grayscale
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    print(f"[✓] Converted to grayscale")
    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    print(f"[✓] Applied Gaussian blur (5x5 kernel)")
    
    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)
    print(f"[✓] Applied adaptive thresholding")
    
    # Display preprocessing results
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Original (512x512)', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    axes[1].imshow(gray, cmap='gray')
    axes[1].set_title('Grayscale', fontsize=12, fontweight='bold')
    axes[1].axis('off')
    
    axes[2].imshow(thresh, cmap='gray')
    axes[2].set_title('Adaptive Thresholding', fontsize=12, fontweight='bold')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig('outputs/01_preprocessing.png', dpi=150, bbox_inches='tight')
    print(f"[✓] Saved: outputs/01_preprocessing.png")
    plt.show()
    
    return img_resized, gray, blurred, thresh


def sampling_analysis(gray_img):
    """
    Perform resolution sampling analysis
    Returns: dictionary of sampled images and PSNR values
    """
    print("\n[TASK 2] Sampling Analysis (Resolution)")
    print("-" * 60)
    
    resolutions = {
        'High (512x512)': 512,
        'Medium (256x256)': 256,
        'Low (128x128)': 128
    }
    
    sampled_images = {}
    psnr_values = {}
    
    for name, size in resolutions.items():
        # Downsample
        downsampled = cv2.resize(gray_img, (size, size), interpolation=cv2.INTER_AREA)
        # Upsample back to 512x512
        upsampled = cv2.resize(downsampled, (512, 512), interpolation=cv2.INTER_LINEAR)
        
        sampled_images[name] = upsampled
        
        # Calculate PSNR
        psnr = calculate_psnr(gray_img, upsampled)
        psnr_values[name] = psnr
        
        print(f"[✓] {name}: PSNR = {psnr:.2f} dB")
    
    # Display comparison
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    axes[0].imshow(gray_img, cmap='gray')
    axes[0].set_title('Original\n512x512', fontsize=11, fontweight='bold')
    axes[0].axis('off')
    
    for idx, (name, img) in enumerate(sampled_images.items(), 1):
        axes[idx].imshow(img, cmap='gray')
        psnr = psnr_values[name]
        axes[idx].set_title(f'{name}\nPSNR: {psnr:.2f} dB', fontsize=11, fontweight='bold')
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig('outputs/02_sampling_analysis.png', dpi=150, bbox_inches='tight')
    print(f"[✓] Saved: outputs/02_sampling_analysis.png")
    plt.show()
    
    return sampled_images, psnr_values


def quantization_analysis(gray_img):
    """
    Perform quantization analysis
    Returns: dictionary of quantized images and quality metrics
    """
    print("\n[TASK 3] Quantization Analysis (Gray Level Reduction)")
    print("-" * 60)
    
    quantization_levels = {
        '8-bit (256 levels)': 256,
        '4-bit (16 levels)': 16,
        '2-bit (4 levels)': 4
    }
    
    quantized_images = {}
    mse_values = {}
    psnr_values = {}
    
    for name, levels in quantization_levels.items():
        # Quantize image
        quantized = quantize_image(gray_img, levels)
        quantized_images[name] = quantized
        
        # Calculate MSE and PSNR
        mse = calculate_mse(gray_img, quantized)
        psnr = calculate_psnr(gray_img, quantized)
        
        mse_values[name] = mse
        psnr_values[name] = psnr
        
        print(f"[✓] {name}: MSE = {mse:.2f}, PSNR = {psnr:.2f} dB")
    
    # Display comparison
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    axes[0].imshow(gray_img, cmap='gray')
    axes[0].set_title('Original\n(8-bit)', fontsize=11, fontweight='bold')
    axes[0].axis('off')
    
    for idx, (name, img) in enumerate(quantized_images.items(), 1):
        axes[idx].imshow(img, cmap='gray')
        psnr = psnr_values[name]
        axes[idx].set_title(f'{name}\nPSNR: {psnr:.2f} dB', fontsize=11, fontweight='bold')
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig('outputs/03_quantization_analysis.png', dpi=150, bbox_inches='tight')
    print(f"[✓] Saved: outputs/03_quantization_analysis.png")
    plt.show()
    
    return quantized_images, mse_values, psnr_values


def quantize_image(img, levels):
    """Quantize image to specified number of gray levels"""
    factor = 256 / levels
    quantized = np.floor(img / factor) * factor
    return quantized.astype(np.uint8)


def calculate_mse(img1, img2):
    """Calculate Mean Squared Error"""
    mse = np.mean((img1.astype(float) - img2.astype(float)) ** 2)
    return mse


def calculate_psnr(img1, img2):
    """Calculate Peak Signal-to-Noise Ratio"""
    mse = calculate_mse(img1, img2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr


def ocr_analysis(gray_img, sampled_images, quantized_images):
    """
    Perform OCR analysis on different image qualities
    """
    print("\n[TASK 4] OCR Analysis using Tesseract")
    print("-" * 60)
    
    ocr_results = {}
    
    # Test images
    test_images = {
        'Original Grayscale': gray_img,
        'Medium Resolution (256x256)': sampled_images['Medium (256x256)'],
        'Low Resolution (128x128)': sampled_images['Low (128x128)'],
        '4-bit Quantized': quantized_images['4-bit (16 levels)'],
        '2-bit Quantized': quantized_images['2-bit (4 levels)']
    }
    
    print("\nExtracting text from images...")
    
    for name, img in test_images.items():
        try:
            text = pytesseract.image_to_string(img)
            char_count = len(text.strip())
            ocr_results[name] = {
                'text': text,
                'char_count': char_count
            }
            print(f"[✓] {name}: {char_count} characters extracted")
        except Exception as e:
            ocr_results[name] = {
                'text': f"Error: {str(e)}",
                'char_count': 0
            }
            print(f"[✗] {name}: OCR failed - {str(e)}")
    
    # Save OCR results
    save_ocr_results(ocr_results)
    
    # Print analysis
    print_ocr_observations(ocr_results)
    
    return ocr_results


def save_ocr_results(ocr_results):
    """Save OCR results to text file"""
    with open('outputs/ocr_results.txt', 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("OCR ANALYSIS RESULTS\n")
        f.write("="*80 + "\n\n")
        
        for name, result in ocr_results.items():
            f.write(f"\n{'='*80}\n")
            f.write(f"Image Type: {name}\n")
            f.write(f"Character Count: {result['char_count']}\n")
            f.write(f"{'-'*80}\n")
            f.write(f"Extracted Text:\n{result['text']}\n")
    
    print(f"[✓] Saved: outputs/ocr_results.txt")


def print_ocr_observations(ocr_results):
    """Print structured OCR observations"""
    print("\n" + "="*80)
    print("OCR QUALITY OBSERVATIONS")
    print("="*80)
    
    original_count = ocr_results['Original Grayscale']['char_count']
    
    print("\n1. TEXT CLARITY ANALYSIS:")
    print("   • Original Grayscale: Baseline reference with maximum clarity")
    print("   • Medium Resolution: Slight degradation, most text still readable")
    print("   • Low Resolution: Significant quality loss, text may be blurry")
    print("   • 4-bit Quantized: Visible posterization, reduced contrast")
    print("   • 2-bit Quantized: Severe quality loss, limited gray levels")
    
    print("\n2. READABILITY DEGRADATION:")
    for name, result in ocr_results.items():
        if original_count > 0:
            accuracy = (result['char_count'] / original_count) * 100
        else:
            accuracy = 0
        print(f"   • {name}: {result['char_count']} chars ({accuracy:.1f}% of original)")
    
    print("\n3. OCR SUITABILITY:")
    print("   • High Resolution (512x512): Excellent for OCR")
    print("   • Medium Resolution (256x256): Good for OCR, acceptable accuracy")
    print("   • Low Resolution (128x128): Poor for OCR, significant errors")
    print("   • 8-bit Quantization: Excellent, no visible impact")
    print("   • 4-bit Quantization: Moderate, some accuracy loss")
    print("   • 2-bit Quantization: Poor, not recommended for OCR")
    
    print("\n4. RECOMMENDATIONS:")
    print("   • Minimum resolution: 256x256 for acceptable OCR")
    print("   • Minimum quantization: 4-bit (16 levels) for basic OCR")
    print("   • Optimal: 512x512 resolution with 8-bit quantization")
    print("="*80 + "\n")


def create_comprehensive_comparison(gray_img, sampled_images, quantized_images):
    """Create single comprehensive comparison figure"""
    print("\n[TASK 5] Creating Comprehensive Comparison Figure")
    print("-" * 60)
    
    fig = plt.figure(figsize=(18, 10))
    
    # Original
    ax1 = plt.subplot(2, 4, 1)
    ax1.imshow(gray_img, cmap='gray')
    ax1.set_title('Original\n512x512', fontsize=12, fontweight='bold')
    ax1.axis('off')
    
    # Sampled images
    ax2 = plt.subplot(2, 4, 2)
    ax2.imshow(sampled_images['High (512x512)'], cmap='gray')
    ax2.set_title('High Resolution\n512x512', fontsize=12, fontweight='bold')
    ax2.axis('off')
    
    ax3 = plt.subplot(2, 4, 3)
    ax3.imshow(sampled_images['Medium (256x256)'], cmap='gray')
    ax3.set_title('Medium Resolution\n256x256', fontsize=12, fontweight='bold')
    ax3.axis('off')
    
    ax4 = plt.subplot(2, 4, 4)
    ax4.imshow(sampled_images['Low (128x128)'], cmap='gray')
    ax4.set_title('Low Resolution\n128x128', fontsize=12, fontweight='bold')
    ax4.axis('off')
    
    # Quantized images
    ax5 = plt.subplot(2, 4, 5)
    ax5.imshow(quantized_images['8-bit (256 levels)'], cmap='gray')
    ax5.set_title('8-bit Quantization\n256 levels', fontsize=12, fontweight='bold')
    ax5.axis('off')
    
    ax6 = plt.subplot(2, 4, 6)
    ax6.imshow(quantized_images['4-bit (16 levels)'], cmap='gray')
    ax6.set_title('4-bit Quantization\n16 levels', fontsize=12, fontweight='bold')
    ax6.axis('off')
    
    ax7 = plt.subplot(2, 4, 7)
    ax7.imshow(quantized_images['2-bit (4 levels)'], cmap='gray')
    ax7.set_title('2-bit Quantization\n4 levels', fontsize=12, fontweight='bold')
    ax7.axis('off')
    
    # Histogram
    ax8 = plt.subplot(2, 4, 8)
    ax8.hist(gray_img.ravel(), bins=256, range=[0, 256], color='blue', alpha=0.7)
    ax8.set_title('Histogram\n(Original)', fontsize=12, fontweight='bold')
    ax8.set_xlabel('Pixel Intensity')
    ax8.set_ylabel('Frequency')
    ax8.grid(True, alpha=0.3)
    
    plt.suptitle('Smart Document Scanner - Comprehensive Quality Analysis', 
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig('outputs/04_comprehensive_comparison.png', dpi=150, bbox_inches='tight')
    print(f"[✓] Saved: outputs/04_comprehensive_comparison.png")
    plt.show()


def main():
    """Main execution function"""
    print_welcome()
    create_output_folder()
    
    # Get image path from user
    print("\n📁 Please upload your document image")
    image_path = input("Enter image path: ").strip()
    
    if not image_path or not os.path.exists(image_path):
        print(f"\n[✗] Error: Image not found at '{image_path}'")
        print("Please provide a valid image path.")
        sys.exit(1)
    
    # Task 1: Image Acquisition & Preprocessing
    img_resized, gray, blurred, thresh = load_and_preprocess_image(image_path)
    
    # Task 2: Sampling Analysis
    sampled_images, sampling_psnr = sampling_analysis(gray)
    
    # Task 3: Quantization Analysis
    quantized_images, mse_values, quant_psnr = quantization_analysis(gray)
    
    # Task 4: OCR Analysis
    ocr_results = ocr_analysis(gray, sampled_images, quantized_images)
    
    # Task 5: Comprehensive Comparison
    create_comprehensive_comparison(gray, sampled_images, quantized_images)
    
    # Final summary
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print("\nAll outputs saved in 'outputs/' folder:")
    print("  • 01_preprocessing.png")
    print("  • 02_sampling_analysis.png")
    print("  • 03_quantization_analysis.png")
    print("  • 04_comprehensive_comparison.png")
    print("  • ocr_results.txt")
    print("\nThank you for using Smart Document Scanner!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
