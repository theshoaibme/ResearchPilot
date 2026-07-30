import numpy as np
import scipy.ndimage

def normalize_min_max(img):
    """Linearly rescales image intensity array to [0.0, 1.0]."""
    img_min, img_max = np.min(img), np.max(img)
    if img_max > img_min:
        return ((img - img_min) / (img_max - img_min)).astype(np.float32)
    return np.zeros_like(img, dtype=np.float32)

def apply_clahe_2d(img):
    """Applies Contrast Limited Adaptive Histogram Equalization (CLAHE) for X-Ray / Ultrasound."""
    try:
        import cv2
        img_uint8 = (normalize_min_max(img) * 255).astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        equalized = clahe.apply(img_uint8)
        return (equalized / 255.0).astype(np.float32)
    except ImportError:
        # Fallback if cv2 is not available
        return normalize_min_max(img)

def resize_2d(img, target_size=(512, 512), is_mask=False):
    """Resizes 2D medical image to target resolution (512x512)."""
    h, w = img.shape[:2]
    zoom_factors = (target_size[0] / h, target_size[1] / w)
    order = 0 if is_mask else 1
    resized = scipy.ndimage.zoom(img, zoom_factors, order=order)
    return resized

def ct_hu_windowing(ct_volume, hu_min=-150, hu_max=250):
    """Clips CT volume to target Hounsfield Units and normalizes to [0.0, 1.0]."""
    clipped = np.clip(ct_volume, hu_min, hu_max)
    normalized = (clipped - hu_min) / (hu_max - hu_min)
    return normalized.astype(np.float32)

def mri_zscore_normalize(mri_volume):
    """Applies Z-score normalization to MRI volume (background-masked)."""
    mask = mri_volume > 0
    if not np.any(mask):
        return np.zeros_like(mri_volume, dtype=np.float32)
    mean = np.mean(mri_volume[mask])
    std = np.std(mri_volume[mask])
    normalized = np.zeros_like(mri_volume, dtype=np.float32)
    if std > 0:
        normalized[mask] = (mri_volume[mask] - mean) / std
    return normalized

def resample_volume_3d(volume, current_spacing, target_spacing=(1.5, 1.5, 1.5), is_mask=False):
    """Resamples 3D volume to isotropic target voxel spacing."""
    current_spacing = np.array(current_spacing, dtype=np.float32)
    target_spacing = np.array(target_spacing, dtype=np.float32)
    resize_factor = current_spacing / target_spacing
    new_shape = np.round(volume.shape * resize_factor)
    real_resize_factor = new_shape / volume.shape
    order = 0 if is_mask else 1
    resampled = scipy.ndimage.zoom(volume, real_resize_factor, order=order)
    return resampled
