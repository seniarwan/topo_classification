"""
Simple utility functions
"""

import numpy as np
import matplotlib.pyplot as plt
import rasterio

def plot_results(classification, slope=None, convexity=None, texture=None):
    """
    Plot classification results and metrics (excluding nodata)
    
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
    
    # Create master nodata mask
    if slope is not None:
        nodata_mask = np.isnan(slope) | np.isnan(convexity) | np.isnan(texture)
    else:
        nodata_mask = classification == 0
    
    # Plot classification (mask nodata areas)
    class_masked = np.ma.masked_where(classification == 0, classification)
    im1 = axes[0].imshow(class_masked, cmap='tab20', vmin=1, vmax=24)
    axes[0].set_title('Topographic Classification')
    axes[0].axis('off')
    plt.colorbar(im1, ax=axes[0], shrink=0.8, label='Class')
    
    if len(axes) > 1 and slope is not None:
        # Plot slope (mask nodata)
        slope_masked = np.ma.masked_where(nodata_mask, slope)
        im2 = axes[1].imshow(slope_masked, cmap='viridis')
        axes[1].set_title('Slope (degrees)')
        axes[1].axis('off')
        plt.colorbar(im2, ax=axes[1], shrink=0.8)
        
        # Plot convexity (mask nodata)
        convexity_masked = np.ma.masked_where(nodata_mask, convexity)
        im3 = axes[2].imshow(convexity_masked, cmap='RdBu_r', vmin=-0.1, vmax=0.1)
        axes[2].set_title('Convexity')
        axes[2].axis('off')
        plt.colorbar(im3, ax=axes[2], shrink=0.8)
        
        # Plot texture (mask nodata)
        texture_masked = np.ma.masked_where(nodata_mask, texture)
        im4 = axes[3].imshow(texture_masked, cmap='plasma')
        axes[3].set_title('Texture')
        axes[3].axis('off')
        plt.colorbar(im4, ax=axes[3], shrink=0.8)
    
    plt.tight_layout()
    plt.show()

def save_geotiff(data, output_path, profile):
    """
    Save array as GeoTIFF (with proper nodata handling)
    
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
        'compress': 'lzw',
        'nodata': 0  # Set 0 as nodata for classification
    })
    
    # Ensure nodata areas are set to 0
    output_data = data.copy().astype(np.uint8)
    if np.any(np.isnan(data)):
        output_data[np.isnan(data)] = 0
    
    with rasterio.open(output_path, 'w', **out_profile) as dst:
        dst.write(output_data, 1)
    
    print(f"Saved to: {output_path}")

def create_valid_data_mask(*arrays):
    """
    Create mask for valid (non-nodata) pixels across multiple arrays
    
    Parameters:
    -----------
    *arrays : numpy.ndarray
        Variable number of arrays to check
        
    Returns:
    --------
    numpy.ndarray
        Boolean mask where True = valid data
    """
    valid_mask = np.ones(arrays[0].shape, dtype=bool)
    
    for arr in arrays:
        if arr is not None:
            valid_mask &= ~np.isnan(arr)
            valid_mask &= np.isfinite(arr)
    
    return valid_mask

def plot_results_advanced(classification, slope=None, convexity=None, texture=None, 
                          title_suffix="", figsize=(15, 12)):
    """
    Advanced plotting with better nodata handling and color schemes
    """
    # Debug information
    if slope is not None:
        print(f"Debug info:")
        print(f"  Slope: min={np.nanmin(slope):.2f}, max={np.nanmax(slope):.2f}, valid={np.sum(~np.isnan(slope))}")
        print(f"  Convexity: min={np.nanmin(convexity):.4f}, max={np.nanmax(convexity):.4f}, valid={np.sum(~np.isnan(convexity))}")
        print(f"  Texture: min={np.nanmin(texture):.4f}, max={np.nanmax(texture):.4f}, valid={np.sum(~np.isnan(texture))}")
    
    # Determine subplot layout
    if all(x is not None for x in [slope, convexity, texture]):
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        axes = axes.flatten()
    else:
        fig, axes = plt.subplots(1, 1, figsize=(8, 6))
        axes = [axes]
    
    # Create comprehensive nodata mask
    if slope is not None:
        nodata_mask = (np.isnan(slope) | np.isnan(convexity) | np.isnan(texture) |
                      ~np.isfinite(slope) | ~np.isfinite(convexity) | ~np.isfinite(texture))
    else:
        nodata_mask = classification == 0
    
    # 1. Classification plot
    class_data = classification.copy().astype(float)
    class_data[classification == 0] = np.nan
    class_masked = np.ma.masked_invalid(class_data)
    
    im1 = axes[0].imshow(class_masked, cmap='tab20', vmin=1, vmax=24)
    axes[0].set_title(f'Topographic Classification{title_suffix}')
    axes[0].axis('off')
    cbar1 = plt.colorbar(im1, ax=axes[0], shrink=0.8, label='Class')
    
    if len(axes) > 1 and slope is not None:
        # 2. Slope plot
        slope_data = slope.copy()
        slope_data[nodata_mask] = np.nan
        slope_masked = np.ma.masked_invalid(slope_data)
        
        im2 = axes[1].imshow(slope_masked, cmap='terrain')
        axes[1].set_title('Slope (degrees)')
        axes[1].axis('off')
        plt.colorbar(im2, ax=axes[1], shrink=0.8)
        
        # 3. Convexity plot  
        convexity_data = convexity.copy()
        convexity_data[nodata_mask] = np.nan
        
        # Check if convexity has valid data
        if np.all(np.isnan(convexity_data)):
            # Show message for empty convexity
            axes[2].text(0.5, 0.5, 'No valid\nconvexity data', 
                        ha='center', va='center', transform=axes[2].transAxes,
                        fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
            axes[2].set_title('Convexity (No Data)')
            axes[2].axis('off')
        else:
            convexity_masked = np.ma.masked_invalid(convexity_data)
            
            # Use symmetric colormap for convexity
            conv_abs_max = np.nanmax(np.abs(convexity_data))
            if conv_abs_max > 0:
                im3 = axes[2].imshow(convexity_masked, cmap='RdBu_r', 
                                    vmin=-conv_abs_max, vmax=conv_abs_max)
            else:
                im3 = axes[2].imshow(convexity_masked, cmap='RdBu_r')
            axes[2].set_title('Convexity')
            axes[2].axis('off')
            plt.colorbar(im3, ax=axes[2], shrink=0.8)
        
        # 4. Texture plot
        texture_data = texture.copy()
        texture_data[nodata_mask] = np.nan
        texture_masked = np.ma.masked_invalid(texture_data)
        
        im4 = axes[3].imshow(texture_masked, cmap='plasma')
        axes[3].set_title('Texture')
        axes[3].axis('off')
        plt.colorbar(im4, ax=axes[3], shrink=0.8)
    
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    if slope is not None:
        valid_pixels = ~nodata_mask
        total_valid = np.sum(valid_pixels)
        total_pixels = classification.size
        
        print(f"\nData Summary:")
        print(f"Total pixels: {total_pixels:,}")
        print(f"Valid pixels: {total_valid:,} ({100*total_valid/total_pixels:.1f}%)")
        print(f"Nodata pixels: {total_pixels-total_valid:,} ({100*(total_pixels-total_valid)/total_pixels:.1f}%)")
        
        if np.any(classification > 0):
            classified_pixels = np.sum(classification > 0)
            print(f"Classified pixels: {classified_pixels:,} ({100*classified_pixels/total_valid:.1f}% of valid)")


def get_class_stats(classification, slope, convexity, texture):
    """
    Get statistics for each class (excluding nodata)
    
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
    unique_classes = unique_classes[unique_classes > 0]  # Exclude background/nodata
    
    for class_id in unique_classes:
        mask = classification == class_id
        
        # Additional check to exclude any remaining nodata
        valid_mask = mask & ~np.isnan(slope) & ~np.isnan(convexity) & ~np.isnan(texture)
        
        if np.any(valid_mask):
            stats[int(class_id)] = {
                'pixel_count': int(np.sum(valid_mask)),
                'mean_slope': float(np.nanmean(slope[valid_mask])),
                'mean_convexity': float(np.nanmean(convexity[valid_mask])),
                'mean_texture': float(np.nanmean(texture[valid_mask]))
            }
    
    return stats
