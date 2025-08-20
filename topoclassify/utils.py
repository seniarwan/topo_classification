"""
Simple utility functions
"""

import numpy as np
import matplotlib.pyplot as plt
import rasterio

def plot_results(classification, slope=None, convexity=None, texture=None):
    """
    Plot classification results and metrics
    
    Parameters:
    -----------
    classification : numpy.ndarray
        Classification result
    slope : numpy.ndarray, optional
        Slope values
    convexity : numpy.ndarray, optional  
        Convexity values
    texture : numpy.ndarray, optional
        Texture values
    """
    # Determine subplot layout
    if all(x is not None for x in [slope, convexity, texture]):
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
    else:
        fig, axes = plt.subplots(1, 1, figsize=(8, 6))
        axes = [axes]
    
    # Plot classification
    im1 = axes[0].imshow(classification, cmap='tab20', vmin=1, vmax=24)
    axes[0].set_title('Topographic Classification')
    axes[0].axis('off')
    plt.colorbar(im1, ax=axes[0], shrink=0.8, label='Class')
    
    if len(axes) > 1 and slope is not None:
        # Plot slope
        im2 = axes[1].imshow(slope, cmap='viridis')
        axes[1].set_title('Slope (degrees)')
        axes[1].axis('off')
        plt.colorbar(im2, ax=axes[1], shrink=0.8)
        
        # Plot convexity
        im3 = axes[2].imshow(convexity, cmap='RdBu_r')
        axes[2].set_title('Convexity')
        axes[2].axis('off')
        plt.colorbar(im3, ax=axes[2], shrink=0.8)
        
        # Plot texture
        im4 = axes[3].imshow(texture, cmap='plasma')
        axes[3].set_title('Texture')
        axes[3].axis('off')
        plt.colorbar(im4, ax=axes[3], shrink=0.8)
    
    plt.tight_layout()
    plt.show()

def save_geotiff(data, output_path, profile):
    """
    Save array as GeoTIFF
    
    Parameters:
    -----------
    data : numpy.ndarray
        Data to save
    output_path : str
        Output file path
    profile : dict
        Rasterio profile from original DEM
    """
    # Update profile for output
    out_profile = profile.copy()
    out_profile.update({
        'dtype': 'uint8',
        'count': 1,
        'compress': 'lzw'
    })
    
    with rasterio.open(output_path, 'w', **out_profile) as dst:
        dst.write(data.astype(np.uint8), 1)
    
    print(f"Saved to: {output_path}")

def get_class_stats(classification, slope, convexity, texture):
    """
    Get statistics for each class
    
    Parameters:
    -----------
    classification : numpy.ndarray
        Classification result
    slope : numpy.ndarray
        Slope values
    convexity : numpy.ndarray
        Convexity values  
    texture : numpy.ndarray
        Texture values
        
    Returns:
    --------
    dict
        Statistics for each class
    """
    stats = {}
    unique_classes = np.unique(classification)
    unique_classes = unique_classes[unique_classes > 0]
    
    for class_id in unique_classes:
        mask = classification == class_id
        
        stats[int(class_id)] = {
            'pixel_count': int(np.sum(mask)),
            'mean_slope': float(np.nanmean(slope[mask])),
            'mean_convexity': float(np.nanmean(convexity[mask])),
            'mean_texture': float(np.nanmean(texture[mask]))
        }
    
    return stats
