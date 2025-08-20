"""
Simple Topographic Classification Package
Pure Python implementation of Iwahashi & Pike (2007)
"""

__version__ = "1.0.0"

from .classifier import TopographicClassifier
from .metrics import calculate_slope, calculate_convexity, calculate_texture
from .utils import plot_results, save_geotiff, get_class_stats

def quick_classify(dem_path, output_path=None, show_plot=True):
    """
    Quick topographic classification using ArcPy-equivalent algorithms
    
    Parameters:
    -----------
    dem_path : str
        Path to DEM file
    output_path : str, optional
        Path to save result
    show_plot : bool
        Show visualization
        
    Returns:
    --------
    numpy.ndarray
        Classification result (24 classes)
    """
    print("Starting ArcPy-equivalent topographic classification...")
    
    # Create classifier with ArcPy methods
    classifier = TopographicClassifier(dem_path)
    
    # Run classification
    result = classifier.classify()
    
    # Save if requested
    if output_path:
        save_geotiff(result, output_path, classifier.profile)
        print(f"Result saved to: {output_path}")
    
    # Show plot if requested
    if show_plot:
        print("Creating visualization...")
        plot_results(result, classifier.slope, classifier.convexity, classifier.texture)
    
    return result

def classify_detailed(dem_path, output_dir="./output/"):
    """
    Detailed classification with all outputs and statistics
    
    Parameters:
    -----------
    dem_path : str
        Path to DEM file
    output_dir : str
        Output directory for all results
        
    Returns:
    --------
    dict
        Complete results including classification, metrics, and statistics
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    print("Starting detailed ArcPy-equivalent classification...")
    
    # Initialize classifier
    classifier = TopographicClassifier(dem_path)
    
    # Get all metrics
    slope = classifier.slope
    convexity = classifier.convexity  
    texture = classifier.texture
    
    # Perform classification
    classification = classifier.classify()
    
    # Save all outputs
    save_geotiff(classification, f"{output_dir}/classification.tif", classifier.profile)
    save_geotiff((slope * 100).astype('uint16'), f"{output_dir}/slope_x100.tif", classifier.profile)
    save_geotiff((convexity * 1000).astype('int16'), f"{output_dir}/convexity_x1000.tif", classifier.profile)
    save_geotiff((texture * 1000).astype('uint16'), f"{output_dir}/texture_x1000.tif", classifier.profile)
    
    # Calculate statistics
    stats = get_class_stats(classification, slope, convexity, texture)
    descriptions = classifier.get_class_descriptions()
    
    # Save statistics
    _save_statistics(stats, descriptions, f"{output_dir}/statistics.csv")
    
    # Create visualization
    plot_results(classification, slope, convexity, texture)
    
    print(f"All outputs saved to: {output_dir}")
    
    return {
        'classification': classification,
        'slope': slope,
        'convexity': convexity,
        'texture': texture,
        'statistics': stats,
        'descriptions': descriptions,
        'classifier': classifier
    }

def _save_statistics(stats, descriptions, output_path):
    """Save class statistics to CSV"""
    import pandas as pd
    
    stats_data = []
    for class_id in sorted(stats.keys()):
        stat = stats[class_id]
        stats_data.append({
            'Class': class_id,
            'Description': descriptions.get(class_id, 'Unknown'),
            'Pixel_Count': stat['pixel_count'],
            'Mean_Slope': round(stat['mean_slope'], 2),
            'Mean_Convexity': round(stat['mean_convexity'], 4),
            'Mean_Texture': round(stat['mean_texture'], 4)
        })
    
    df = pd.DataFrame(stats_data)
    df.to_csv(output_path, index=False)
    print(f"Statistics saved to: {output_path}")

__all__ = [
    'TopographicClassifier',
    'quick_classify', 
    'classify_detailed',
    'calculate_slope',
    'calculate_convexity', 
    'calculate_texture',
    'plot_results',
    'save_geotiff',
    'get_class_stats'
]
