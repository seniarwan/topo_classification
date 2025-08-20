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
    
    # Apply filter, handle NaN properly
    laplacian = ndimage.convolve(dem, kernel, mode='constant', cval=0.0)
    
    # Handle areas where original DEM was NaN
    dem_nan_mask = np.isnan(dem)
    laplacian[dem_nan_mask] = np.nan
    
    # Binary: convex areas (positive values)
    binary = np.zeros_like(laplacian)
    valid_mask = ~np.isnan(laplacian)
    binary[valid_mask] = (laplacian[valid_mask] > 0).astype(float)
    
    # Set NaN areas to NaN in binary
    binary[dem_nan_mask] = np.nan
    
    # Focal mean with circular window
    selem = disk(10)
    
    # Only process if we have valid data
    if not np.all(np.isnan(binary)):
        try:
            # Convert to uint8 for rank filter, handling NaN
            binary_for_rank = np.zeros_like(binary, dtype=np.uint8)
            valid_binary = ~np.isnan(binary)
            binary_for_rank[valid_binary] = (binary[valid_binary] * 255).astype(np.uint8)
            
            # Add padding to handle edges better
            pad_width = 10
            padded = np.pad(binary_for_rank, pad_width, mode='reflect')
            
            # Apply rank filter
            convexity_padded = rank.mean(padded, selem) / 255.0
            
            # Remove padding
            convexity = convexity_padded[pad_width:-pad_width, pad_width:-pad_width]
            
            # Restore NaN areas
            convexity[dem_nan_mask] = np.nan
            
        except Exception as e:
            print(f"Warning: Rank filter failed ({e}), using convolution fallback")
            # Fallback to convolution
            conv_kernel = np.ones((21, 21)) / (21*21)
            convexity = ndimage.convolve(binary, conv_kernel, mode='constant', cval=0.0)
            convexity[dem_nan_mask] = np.nan
    else:
        print("Warning: All convexity values are NaN")
        convexity = np.full_like(dem, np.nan)
    
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
    
    # Handle NaN in median result
    dem_nan_mask = np.isnan(dem)
    median_dem[dem_nan_mask] = np.nan
    
    # Calculate differences
    diff_pos = np.maximum(0, dem - median_dem)
    diff_neg = np.maximum(0, median_dem - dem)
    texture_raw = diff_pos + diff_neg
    
    # Handle edge case where all values are the same or NaN
    valid_mask = ~np.isnan(texture_raw)
    if not np.any(valid_mask):
        return np.full_like(dem, np.nan)
        
    max_val = np.nanmax(texture_raw)
    if max_val == 0 or np.isnan(max_val):
        result = np.zeros_like(texture_raw)
        result[dem_nan_mask] = np.nan
        return result
    
    # Normalize for rank filter
    texture_norm = np.zeros_like(texture_raw, dtype=np.uint8)
    texture_norm[valid_mask] = (texture_raw[valid_mask] / max_val * 255).astype(np.uint8)
    
    # Focal mean
    selem = disk(10)
    try:
        # Add padding to handle edges better
        pad_width = 10
        padded = np.pad(texture_norm, pad_width, mode='reflect')
        texture_padded = rank.mean(padded, selem) / 255.0
        texture = texture_padded[pad_width:-pad_width, pad_width:-pad_width]
        
        # Restore NaN areas
        texture[dem_nan_mask] = np.nan
        
    except Exception as e:
        print(f"Warning: Rank filter failed for texture ({e}), using convolution fallback")
        # Fallback to convolution
        conv_kernel = np.ones((21, 21)) / (21*21)
        texture = ndimage.convolve(texture_norm.astype(float), conv_kernel, mode='constant') / 255.0
        texture[dem_nan_mask] = np.nan
    
    return texture
