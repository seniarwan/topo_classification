"""
Enhanced Topographic Classifier - Full Iwahashi & Pike (2007) Implementation
"""

import numpy as np
import rasterio
from .metrics import calculate_slope, calculate_convexity, calculate_texture

class TopographicClassifier:
    """
    Main topographic classifier using ArcPy-equivalent algorithms
    """
    
    def __init__(self, dem_path):
        """
        Initialize classifier with ArcPy-equivalent settings
        
        Parameters:
        -----------
        dem_path : str
            Path to DEM file
        """
        self.dem_path = dem_path
        self.load_dem()
        
        # Extract cell size from DEM (like ArcPy)
        if hasattr(self, 'profile') and 'transform' in self.profile:
            self.cell_size = abs(self.profile['transform'][0])
        else:
            self.cell_size = 1.0
        
        self.z_factor = 1.0
        
    def load_dem(self):
        """Load DEM with integer conversion (like ArcPy Int_sa)"""
        with rasterio.open(self.dem_path) as src:
            dem_raw = src.read(1)
            self.profile = src.profile
            
            # Handle nodata
            if src.nodata is not None:
                dem_raw = dem_raw.astype(np.float32)
                dem_raw[dem_raw == src.nodata] = np.nan
            
            # Convert to integer like ArcPy's Int_sa operation
            self.dem_data = np.round(dem_raw).astype(np.float32)
            
            print(f"DEM loaded: {self.dem_data.shape}, Cell size: {self.cell_size}")
    
    def calculate_slope_horn(self):
        """
        Calculate slope using Horn's method (same as ArcGIS Slope tool)
        """
        # Horn's 3rd-order finite difference method - exact ArcGIS implementation
        kernel_x = np.array([[-1, 0, 1],
                            [-2, 0, 2], 
                            [-1, 0, 1]]) / (8.0 * self.cell_size)
        
        kernel_y = np.array([[-1, -2, -1],
                            [0, 0, 0],
                            [1, 2, 1]]) / (8.0 * self.cell_size)
        
        # Apply convolution
        from scipy import ndimage
        dx = ndimage.convolve(self.dem_data, kernel_x, mode='constant', cval=np.nan)
        dy = ndimage.convolve(self.dem_data, kernel_y, mode='constant', cval=np.nan)
        
        # Calculate slope
        slope_rad = np.arctan(self.z_factor * np.sqrt(dx**2 + dy**2))
        slope_deg = np.degrees(slope_rad)
        
        return slope_deg
    
    def calculate_convexity_arcpy(self):
        """
        Calculate convexity exactly following ArcPy workflow
        """
        # Step 1: Apply exact Laplacian filter (FocalStatistics with Weight)
        kernel = np.array([[-1, -1, -1],
                          [-1,  8, -1],
                          [-1, -1, -1]], dtype=np.float32)
        
        from scipy import ndimage
        laplacian = ndimage.convolve(self.dem_data, kernel, mode='constant', cval=0.0)
        
        # Handle nodata
        dem_nan_mask = np.isnan(self.dem_data)
        laplacian[dem_nan_mask] = np.nan
        
        # Step 2: Con operation (VALUE > 0 -> 1, else 0)
        binary = np.zeros_like(laplacian)
        valid_mask = ~np.isnan(laplacian)
        binary[valid_mask & (laplacian > 0)] = 1.0
        binary[~valid_mask] = np.nan
        
        # Step 3: FocalStatistics with Circle 10 CELL, MEAN
        convexity = self._focal_statistics_circular(binary, radius=10)
        
        return convexity
    
    def calculate_texture_arcpy(self):
        """
        Calculate texture exactly following ArcPy workflow
        """
        from scipy import ndimage
        
        # Step 1: Median filter (3x3 rectangle) 
        median_dem = ndimage.median_filter(self.dem_data, size=3)
        
        # Handle NaN
        dem_nan_mask = np.isnan(self.dem_data)
        median_dem[dem_nan_mask] = np.nan
        
        # Step 2: Minus operations
        diff_pos = np.maximum(0, self.dem_data - median_dem)  # Con operation
        diff_neg = np.maximum(0, median_dem - self.dem_data)  # Con operation
        
        # Step 3: Plus operation
        texture_raw = diff_pos + diff_neg
        
        # Step 4: Float operation
        texture_float = texture_raw.astype(np.float32)
        
        # Step 5: FocalStatistics Circle 10 CELL, MEAN
        texture = self._focal_statistics_circular(texture_float, radius=10)
        
        return texture
    
    def _focal_statistics_circular(self, data, radius=10):
        """
        Implement exact circular focal statistics like ArcPy
        """
        # Create exact circular kernel
        size = 2 * radius + 1
        y, x = np.ogrid[-radius:radius+1, -radius:radius+1]
        circle_mask = x**2 + y**2 <= radius**2
        
        # Initialize result
        result = np.full_like(data, np.nan)
        
        # Apply focal statistics with circular window
        for i in range(radius, data.shape[0] - radius):
            for j in range(radius, data.shape[1] - radius):
                if not np.isnan(data[i, j]):
                    # Extract circular neighborhood
                    neighborhood = data[i-radius:i+radius+1, j-radius:j+radius+1]
                    circular_values = neighborhood[circle_mask]
                    
                    # Calculate mean of valid values
                    valid_values = circular_values[~np.isnan(circular_values)]
                    if len(valid_values) > 0:
                        result[i, j] = np.mean(valid_values)
        
        return result
    
    def create_iterative_masks(self, slope):
        """
        Create iterative masks following exact ArcPy zonal statistics workflow
        """
        masks = []
        current_mask = ~np.isnan(slope)
        
        # Follow exact ArcPy sequence: ZonalStatistics -> GreaterThan -> SetNull
        for iteration in range(5):
            if not np.any(current_mask):
                masks.append(np.zeros_like(slope, dtype=bool))
                continue
            
            # ZonalStatistics MEAN
            mean_slope = np.nanmean(slope[current_mask])
            
            # GreaterThan operation
            slope_gt_mean = (slope > mean_slope) & current_mask
            
            # SetNull operation (exclude areas with slope > mean)
            next_mask = current_mask & ~slope_gt_mean
            
            masks.append(slope_gt_mean.copy())
            current_mask = next_mask
        
        return masks
    
    @property
    def slope(self):
        """Calculate slope using Horn's method (cached)"""
        if not hasattr(self, '_slope'):
            print("Calculating slope (Horn's method)...")
            self._slope = self.calculate_slope_horn()
        return self._slope
    
    @property 
    def convexity(self):
        """Calculate convexity using ArcPy method (cached)"""
        if not hasattr(self, '_convexity'):
            print("Calculating convexity (ArcPy method)...")
            self._convexity = self.calculate_convexity_arcpy()
        return self._convexity
    
    @property
    def texture(self):
        """Calculate texture using ArcPy method (cached)"""
        if not hasattr(self, '_texture'):
            print("Calculating texture (ArcPy method)...")
            self._texture = self.calculate_texture_arcpy()
        return self._texture
    
    def classify(self):
        """
        Perform classification following exact ArcPy workflow sequence
        """
        print("Starting topographic classification...")
        
        # Get all metrics using ArcPy methods
        slope = self.slope
        convexity = self.convexity
        texture = self.texture
        
        # Create iterative masks
        print("Creating hierarchical masks...")
        slope_masks = self.create_iterative_masks(slope)
        
        # Initialize classification
        classification = np.zeros_like(slope, dtype=np.uint8)
        
        print("Performing hierarchical classification...")
        
        # Process 6 levels following exact ArcPy sequence
        for level in range(6):
            if level == 0:
                # Level 0: entire area (classes 1-4)
                current_mask = ~np.isnan(slope) & ~np.isnan(convexity) & ~np.isnan(texture)
                base_class = 1
            else:
                # Levels 1-5: iterative masks (classes 5-8, 9-12, 13-16, 17-20, 21-24)
                if level-1 < len(slope_masks):
                    current_mask = slope_masks[level-1]
                else:
                    continue
                base_class = 1 + (level * 4)
            
            if not np.any(current_mask):
                continue
            
            # ZonalStatistics for current mask
            slope_mean = np.nanmean(slope[current_mask])
            conv_mean = np.nanmean(convexity[current_mask])
            text_mean = np.nanmean(texture[current_mask])
            
            # Skip if thresholds invalid
            if np.isnan(slope_mean) or np.isnan(conv_mean) or np.isnan(text_mean):
                continue
            
            # GreaterThan operations
            s_high = (slope > slope_mean) & current_mask
            c_high = (convexity > conv_mean) & current_mask
            t_high = (texture > text_mean) & current_mask
            
            # BooleanAnd operations - 4 combinations per level
            # Following exact ArcPy workflow pattern
            
            # Class X+0: s=1, c=1, t=1
            mask = s_high & c_high & t_high
            classification[mask] = base_class
            
            # Class X+1: s=1, c=1, t=0
            mask = s_high & c_high & (~t_high & current_mask)
            classification[mask] = base_class + 1
            
            # Class X+2: s=1, c=0, t=1  
            mask = s_high & (~c_high & current_mask) & t_high
            classification[mask] = base_class + 2
            
            # Class X+3: s=1, c=0, t=0
            mask = s_high & (~c_high & current_mask) & (~t_high & current_mask)
            classification[mask] = base_class + 3
        
        # Post-processing: Nibble operation
        self._apply_nibble(classification)
        
        # Get final statistics
        unique_classes = np.unique(classification)
        unique_classes = unique_classes[unique_classes > 0]
        print(f"Classification completed! Classes: {sorted(unique_classes)}")
        
        return classification
    
    def _apply_nibble(self, classification):
        """
        Apply nibble operation for unclassified pixels (like ArcPy Nibble)
        """
        unclassified = (classification == 0) & (~np.isnan(self.dem_data))
        
        if np.any(unclassified):
            print(f"Applying nibble to {np.sum(unclassified)} unclassified pixels...")
            
            from scipy.ndimage import maximum_filter
            
            # Iterative dilation to fill gaps
            for iteration in range(3):
                dilated = maximum_filter(classification, size=3)
                classification[unclassified] = dilated[unclassified]
                unclassified = (classification == 0) & (~np.isnan(self.dem_data))
                if not np.any(unclassified):
                    break
    
    def get_class_descriptions(self):
        """
        Get descriptions for each class following Iwahashi & Pike methodology
        """
        descriptions = {
            1: "Very steep, highly convex, very rough",
            2: "Very steep, highly convex, smooth", 
            3: "Very steep, concave, very rough",
            4: "Very steep, concave, smooth",
            5: "Steep, highly convex, very rough",
            6: "Steep, highly convex, smooth",
            7: "Steep, concave, very rough", 
            8: "Steep, concave, smooth",
            9: "Moderately steep, highly convex, very rough",
            10: "Moderately steep, highly convex, smooth",
            11: "Moderately steep, concave, very rough",
            12: "Moderately steep, concave, smooth",
            13: "Gentle, highly convex, very rough",
            14: "Gentle, highly convex, smooth",
            15: "Gentle, concave, very rough",
            16: "Gentle, concave, smooth", 
            17: "Very gentle, highly convex, very rough",
            18: "Very gentle, highly convex, smooth",
            19: "Very gentle, concave, very rough",
            20: "Very gentle, concave, smooth",
            21: "Nearly flat, highly convex, very rough",
            22: "Nearly flat, highly convex, smooth",
            23: "Nearly flat, concave, very rough", 
            24: "Nearly flat, concave, smooth"
        }
        return descriptions
