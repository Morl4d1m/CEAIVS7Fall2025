import cv2
import os
import torch
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
import time

"""
Requirements: (libraries and modules)
pip uninstall numpy (RealESRGANer can only run on older version of numpy)
pip install numpy==1.26.4
pip install torch==2.0.1 torchvision==0.15.2 (Same goes for torch and torchvision)
pip install basicsr facexlib gfpgan realesrgan
pip install time

Make sure to change the paths below, so it fits the directory
"""

# Start Timer:
start_t = time.time()

# Define paths
img_path = 'Super-resolutionCoding\\survelliancepic.jpg'
edsr_model_path = 'Super-resolutionCoding\\EDSR_x4.pb'  # EDSR model (x4)
real_esrgan_model_path = 'Super-resolutionCoding\\RealESRGAN_x4plus.pth'  # Real-ESRGAN model (x4)

# Load image
img = cv2.imread(img_path)
if img is None:
    print(f"Error: Could not load image at {img_path}. Check file path or name.")
    print(f"Current directory: {os.getcwd()}")
    print(f"Files in Super-resolutionCoding: {os.listdir('Super-resolutionCoding')}")
    exit()

# Get low-res dimensions (target output size)
target_height, target_width = img.shape[:2]
print(f"Low-res input shape: {target_width}x{target_height}")

# Step 2: Verify models exist
if not os.path.exists(edsr_model_path):
    print(f"Error: EDSR model not found at {edsr_model_path}. Download from https://github.com/Saafke/EDSR_Tensorflow/tree/master/models")
    exit()
if not os.path.exists(real_esrgan_model_path):
    print(f"Error: Real-ESRGAN model not found at {real_esrgan_model_path}. Download from https://github.com/xinntao/Real-ESRGAN/releases")
    exit()

# Step 3: Initialize EDSR model
sr = cv2.dnn_superres.DnnSuperResImpl_create()
try:
    sr.readModel(edsr_model_path)
    scale = 4  # For EDSR_x4.pb
    sr.setModel("edsr", scale)
except cv2.error as e:
    print(f"Error loading EDSR model: {e}")
    exit()

# Initialize Real-ESRGAN model
model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
upsampler = RealESRGANer(
    scale=4,
    model_path=real_esrgan_model_path,
    model=model,
    tile=0,
    tile_pad=10,
    pre_pad=0,
    half=False  # Set to True if using GPU with ROCm
)

# Step 4: Upscale with EDSR and downscale back to original size with bicubic
edsr_upscaled = sr.upsample(img)
print(f"EDSR upscaled shape: {edsr_upscaled.shape[1]}x{edsr_upscaled.shape[0]}")
#edsr_image = cv2.resize(edsr_upscaled, (target_width, target_height), interpolation=cv2.INTER_CUBIC)
print(f"EDSR final shape: {edsr_upscaled.shape[1]}x{edsr_upscaled.shape[0]}")

# Step 5: Upscale with Real-ESRGAN and downscale back
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for PyTorch
real_esrgan_upscaled, _ = upsampler.enhance(img_rgb, outscale=4)
real_esrgan_upscaled = cv2.cvtColor(real_esrgan_upscaled, cv2.COLOR_RGB2BGR)  # Convert back to BGR
print(f"Real-ESRGAN upscaled shape: {real_esrgan_upscaled.shape[1]}x{real_esrgan_upscaled.shape[0]}")
#real_esrgan_image = cv2.resize(real_esrgan_upscaled, (target_width, target_height), interpolation=cv2.INTER_CUBIC)
print(f"Real-ESRGAN final shape: {real_esrgan_upscaled.shape[1]}x{real_esrgan_upscaled.shape[0]}")

# Step 6: Compare with bicubic (upscale 4x and downscale back)
bicubic_upscaled = cv2.resize(img, (target_width * scale, target_height * scale), interpolation=cv2.INTER_CUBIC)
print(f"Bicubic final shape: {bicubic_upscaled.shape[1]}x{bicubic_upscaled.shape[0]}")

# Step 7: Save results
output_dir = 'SavedPic'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)  # Create SavedPic folder
cv2.imwrite(r'SavedPic\edsr_enhanced.jpg', edsr_upscaled)
cv2.imwrite(r'SavedPic\real_esrgan_enhanced.jpg', real_esrgan_upscaled)
cv2.imwrite(r'SavedPic\bicubic_enhanced.jpg', bicubic_upscaled)
print("Images saved as 'edsr_enhanced.jpg', 'real_esrgan_enhanced.jpg', and 'bicubic_enhanced.jpg'")

# Step 8: Optional display
print("Processing images ...")
cv2.imshow('Low-Res Input', img)
cv2.imshow('EDSR Enhanced', edsr_upscaled)
cv2.imshow('Real-ESRGAN Enhanced', real_esrgan_upscaled)
cv2.imshow('Bicubic Enhanced', bicubic_upscaled)
print("Images DONE!\n")
slut_t = time.time() - start_t
print(f"Endelig tid {slut_t} s")
cv2.waitKey(0)
cv2.destroyAllWindows()
