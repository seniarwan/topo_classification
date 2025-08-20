"""
Simple Topographic Classification Package
Pure Python implementation of Iwahashi & Pike (2007)
"""

__version__ = "1.0.0"

from .classifier import TopographicClassifier
from .metrics import calculate_slope, calculate_convexity, calculate_texture
from .utils import plot_results, save_geotiff, plot_results_advanced

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
    print("Starting quick classification...")
    
    # Create classifier - SAME AS MANUAL
    classifier = TopographicClassifier(dem_path)
    
    # Run classification - SAME AS MANUAL
    result = classifier.classify()
    
    # Save if requested
    if output_path:
        save_geotiff(result, output_path, classifier.profile)
        print(f"Result saved to: {output_path}")
    
    # Show plot if requested - USE STANDARD FUNCTION (same as manual)
    if show_plot:
        print("Creating visualization...")
        plot_results(result, classifier.slope, classifier.convexity, classifier.texture)
    
    return result

def test_consistency(dem_path):
    """
    Test function to verify quick_classify and manual produce same results
    
    Parameters:
    -----------
    dem_path : str
        Path to DEM file
        
    Returns:
    --------
    bool
        True if results are consistent
    """
    print("🧪 Testing consistency between quick_classify and manual classification...")
    
    # Quick classify
    print("\n1. Running quick_classify...")
    quick_result = quick_classify(dem_path, show_plot=False)
    
    # Manual classification
    print("\n2. Running manual classification...")
    classifier = TopographicClassifier(dem_path)
    manual_result = classifier.classify()
    
    # Compare
    from .utils import compare_results
    is_consistent = compare_results(manual_result, quick_result, "Manual", "Quick")
    
    if is_consistent:
        print("\n✅ PASS: Both methods produce identical results!")
    else:
        print("\n❌ FAIL: Results differ between methods!")
    
    return is_consistent

__all__ = ['TopographicClassifier', 'quick_classify', 'test_consistency', 'calculate_slope', 
           'calculate_convexity', 'calculate_texture', 'plot_results', 'plot_results_advanced', 'save_geotiff']
