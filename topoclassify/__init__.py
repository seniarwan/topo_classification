"""
Topographic Classification Package
Python implementation of Iwahashi & Pike (2007)
"""

__version__ = "1.0.0"

from .classifier import TopographicClassifier
from .metrics import calculate_slope, calculate_convexity, calculate_texture
from .utils import plot_results, save_geotiff

def quick_classify(dem_path, output_path=None, show_plot=True):
    """
    Quick classification function for simple usage
    
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
        Classification result
    """
    # Create classifier
    classifier = TopographicClassifier(dem_path)
    
    # Run classification
    result = classifier.classify()
    
    # Save if requested
    if output_path:
        save_geotiff(result, output_path, classifier.profile)
    
    # Show plot if requested
    if show_plot:
        plot_results(result, classifier.slope, classifier.convexity, classifier.texture)
    
    return result

__all__ = ['TopographicClassifier', 'quick_classify', 'calculate_slope', 
           'calculate_convexity', 'calculate_texture', 'plot_results', 'save_geotiff']
