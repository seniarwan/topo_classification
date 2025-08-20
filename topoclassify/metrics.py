"""
Simple metrics calculation functions
"""

import numpy as np
from scipy import ndimage
from skimage.filters import rank
from skimage.morphology import disk

def calculate_slope(dem):
    """
    Calculate slope in degrees
    
    Parameters:
    -----------
    dem : numpy.ndarray
        Digital elevation model
        
    Returns:
    --------
    numpy.ndarray
        Slope in degrees
    """
    # Calculate gradients
    dy, dx = np.gradient(dem)
    
    # Calculate slope
    slope_rad = np.arctan(np.sqrt(dx**2 + dy**2))
    slope_deg = np.degrees(slope_rad)
    
    return slope_deg

def calculate_convexity(dem):
    """
    Calculate convexity using Laplacian filter
    
    Parameters:
    -----------
    dem : numpy.ndarray
        Digital elevation model
        
    Returns:
    --------
    numpy.ndarray
        Convexity values (0-1)
    """
    # Laplacian kernel
    kernel = np.array([[-1, -1, -1],
                       [-1,  8, -1], 
                       [-1, -1, -1]])
    
    # Apply filter
    laplacian = ndimage.convolve(dem, kernel, mode='constant', cval=0.0)
    
    # Binary: convex areas (positive values)
    binary = (laplacian > 0).astype(np.uint8)
    
    # Focal mean with circular window
    selem = disk(10)
    try:
        # Use rank filter if available
        convexity = rank.mean(binary, selem) / 255.0
    except:
        # Fallback to simple convolution
        conv_kernel = np.ones((21, 21)) / (21*21)  # 21x21 for radius ~10
        convexity = ndimage.convolve(binary.astype(float), conv_kernel, mode='constant')
    
    return convexity

def calculate_texture(dem):
    """
    Calculate texture using median differences
    
    Parameters:
    -----------
    dem : numpy.ndarray
        Digital elevation model
        
    Returns:
    --------
    numpy.ndarray
        Texture values (0-1)
    """
    # Median filter
    median_dem = ndimage.median_filter(dem, size=3)
    
    # Calculate differences
    diff_pos = np.maximum(0, dem - median_dem)
    diff_neg = np.maximum(0, median_dem - dem)
    texture_raw = diff_pos + diff_neg
    
    # Handle edge case where all values are the same
    max_val = np.nanmax(texture_raw)
    if max_val == 0 or np.isnan(max_val):
        return np.zeros_like(texture_raw)
    
    # Normalize for rank filter
    texture_norm = (texture_raw / max_val * 255).astype(np.uint8)
    
    # Focal mean
    selem = disk(10)
    try:
        # Add padding to handle edges better
        padded = np.pad(texture_norm, 10, mode='reflect')
        texture_padded = rank.mean(padded, selem) / 255.0
        texture = texture_padded[10:-10, 10:-10]
    except:
        # Fallback to convolution
        conv_kernel = np.ones((21, 21)) / (21*21)
        texture = ndimage.convolve(texture_norm.astype(float), conv_kernel, mode='constant') / 255.0
    
    return texture
