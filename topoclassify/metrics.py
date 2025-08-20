"""
Simple metrics calculation functions
"""

import numpy as np
from scipy import ndimage
from skimage.filters import rank
from skimage.morphology import disk

def calculate_slope(dem, cell_size=1.0, method='horn'):
    """
    Calculate slope using Horn's method
    
    Parameters:
    -----------
    dem : numpy.ndarray
        Digital elevation model
    cell_size : float
        Cell size in map units
    method : str
        Method ('horn')
        
    Returns:
    --------
    numpy.ndarray
        Slope in degrees
    """
    if method == 'horn':
        # Horn's method - same as ArcGIS Slope tool
        kernel_x = np.array([[-1, 0, 1],
                            [-2, 0, 2], 
                            [-1, 0, 1]]) / (8.0 * cell_size)
        
        kernel_y = np.array([[-1, -2, -1],
                            [0, 0, 0],
                            [1, 2, 1]]) / (8.0 * cell_size)
        
        dx = ndimage.convolve(dem, kernel_x, mode='constant', cval=np.nan)
        dy = ndimage.convolve(dem, kernel_y, mode='constant', cval=np.nan)
        
        slope_rad = np.arctan(np.sqrt(dx**2 + dy**2))
        
    else:
        # Simple gradient method (fallback)
        dy, dx = np.gradient(dem)
        slope_rad = np.arctan(np.sqrt(dx**2 + dy**2) / cell_size)
    
    # Convert to degrees
    slope_deg = np.degrees(slope_rad)
    
    return slope_deg

def calculate_convexity(dem, laplacian_kernel=None, focal_radius=10):
    """
    Calculate convexity
    
    Parameters:
    -----------
    dem : numpy.ndarray
        Digital elevation model
    laplacian_kernel : numpy.ndarray, optional
        Laplacian kernel (uses ArcPy default if None)
    focal_radius : int
        Radius for circular focal statistics
        
    Returns:
    --------
    numpy.ndarray
        Convexity values (0-1)
    """
    if laplacian_kernel is None:
        # Exact kernel from ArcPy Laplacian_Filter.txt
        laplacian_kernel = np.array([[-1, -1, -1],
                                   [-1,  8, -1],
                                   [-1, -1, -1]], dtype=np.float32)
    
    # Step 1: Apply Laplacian filter (FocalStatistics with Weight)
    laplacian = ndimage.convolve(dem, laplacian_kernel, mode='constant', cval=0.0)
    
    # Handle nodata properly
    dem_nan_mask = np.isnan(dem)
    laplacian[dem_nan_mask] = np.nan
    
    # Step 2: Con operation (VALUE > 0 -> 1, else 0)
    binary = np.zeros_like(laplacian)
    valid_mask = ~np.isnan(laplacian)
    binary[valid_mask & (laplacian > 0)] = 1.0
    binary[~valid_mask] = np.nan
    
    # Step 3: FocalStatistics Circle MEAN
    convexity = _focal_statistics_circular(binary, radius=focal_radius)
    
    return convexity

def calculate_texture(dem, median_window=3, focal_radius=10):
    """
    Calculate texture using exact ArcPy workflow
    
    Parameters:
    -----------
    dem : numpy.ndarray
        Digital elevation model
    median_window : int
        Window size for median filter
    focal_radius : int
        Radius for circular focal statistics
        
    Returns:
    --------
    numpy.ndarray
        Texture values
    """
    # Step 1: Median filter (Rectangle 3 3 CELL)
    median_dem = ndimage.median_filter(dem, size=median_window)
    
    # Handle NaN
    dem_nan_mask = np.isnan(dem)
    median_dem[dem_nan_mask] = np.nan
    
    # Step 2: Minus operations (like ArcPy workflow)
    diff_pos = np.maximum(0, dem - median_dem)      # TC_dm_md_c
    diff_neg = np.maximum(0, median_dem - dem)      # TC_md_dm_c
    
    # Step 3: Plus operation
    texture_raw = diff_pos + diff_neg               # TC_plus
    
    # Step 4: Float operation
    texture_float = texture_raw.astype(np.float32)  # TC_plus_c_f
    
    # Step 5: FocalStatistics Circle MEAN
    texture = _focal_statistics_circular(texture_float, radius=focal_radius)
    
    return texture

def _focal_statistics_circular(data, radius=10, statistic='mean'):
    """
    Implement circular focal statistics exactly like ArcPy
    
    Parameters:
    -----------
    data : numpy.ndarray
        Input data
    radius : int
        Radius in cells
    statistic : str
        Statistic to calculate ('mean', 'sum', etc.)
        
    Returns:
    --------
    numpy.ndarray
        Result of focal statistics
    """
    # Create exact circular kernel like ArcPy
    size = 2 * radius + 1
    y, x = np.ogrid[-radius:radius+1, -radius:radius+1]
    circle_mask = x**2 + y**2 <= radius**2
    
    # Initialize result
    result = np.full_like(data, np.nan)
    
    # Apply focal statistics
    for i in range(radius, data.shape[0] - radius):
        for j in range(radius, data.shape[1] - radius):
            if not np.isnan(data[i, j]):
                # Extract circular neighborhood
                neighborhood = data[i-radius:i+radius+1, j-radius:j+radius+1]
                circular_values = neighborhood[circle_mask]
                
                # Remove NaN values
                valid_values = circular_values[~np.isnan(circular_values)]
                
                if len(valid_values) > 0:
                    if statistic == 'mean':
                        result[i, j] = np.mean(valid_values)
                    elif statistic == 'sum':
                        result[i, j] = np.sum(valid_values)
                    elif statistic == 'median':
                        result[i, j] = np.median(valid_values)
                    elif statistic == 'std':
                        result[i, j] = np.std(valid_values)
    
    return result

def calculate_all_metrics_arcpy(dem, cell_size=1.0):
    """
    Calculate all metrics
    
    Parameters:
    -----------
    dem : numpy.ndarray
        Digital elevation model
    cell_size : float
        Cell size in map units
        
    Returns:
    --------
    tuple
        (slope, convexity, texture) arrays
    """
    print("Calculating all metrics ...")
    
    # Convert to integer like ArcPy Int_sa
    dem_int = np.round(dem).astype(np.float32)
    
    # Calculate metrics
    slope = calculate_slope(dem_int, cell_size, method='horn')
    convexity = calculate_convexity(dem_int)
    texture = calculate_texture(dem_int)
    
    return slope, convexity, texture
