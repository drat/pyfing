import numpy as np
import cv2 as cv
from .definitions import Image
from .segmentation import SegmentationParameters, SegmentationAlgorithm


class GradMagSegmentationParameters(SegmentationParameters):  
    """
    Parameters of the GradMag segmentation algorithm.    
    """
    
    def __init__(self, window_size = 11, percentile = 95, threshold = 0.2, closing_count = 6, opening_count = 12):
        """
        TODO ...
        """
        self.window_size = window_size
        self.percentile = percentile
        self.threshold = threshold
        self.closing_count = closing_count
        self.opening_count = opening_count


class GradMagSegmentationAlgorithm(SegmentationAlgorithm):
    """
    TODO
    """

    def __init__(self, parameters: GradMagSegmentationParameters = GradMagSegmentationParameters()):
        super().__init__(parameters)
    
    def run(self, image: Image, intermediate_results = None) -> Image:
        parameters = self.parameters
        
        # Calculates gradient magnitude    
        gx, gy = cv.spatialGradient(image)
        sm = cv.magnitude(gx.astype(np.float32), gy.astype(np.float32))
        if intermediate_results is not None: intermediate_results.append((sm, 'Gradient magnitude'))
        
        # Integral on parameters.window_size x parameters.window_size neighborhood
        r = cv.boxFilter(sm, -1, (parameters.window_size, parameters.window_size), normalize = False)
        if intermediate_results is not None: intermediate_results.append((r, 'Sum over the window'))
        
        # Apply the threshold to the gradient magnitude at the required percentile (integrated over the window)
        norm_t = np.percentile(sm, parameters.percentile) * parameters.window_size * parameters.window_size * parameters.threshold
        
        # Selects pixels with gradient magnitude above the normalized threshold
        mask = cv.threshold(r, norm_t, 255, cv.THRESH_BINARY)[1].astype(np.uint8)
        if intermediate_results is not None: intermediate_results.append((np.copy(mask), 'Thresholding'))
        
        if parameters.closing_count > 0:
            # Applies closing to fill small holes and gulfs
            mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, self._se3x3, iterations = parameters.closing_count)        
            if intermediate_results is not None: intermediate_results.append((np.copy(mask), 'After closing'))
            
        # Remove all but the largest component
        self._remove_other_cc(mask)
        if intermediate_results is not None: intermediate_results.append((np.copy(mask), 'Largest component'))
        
        # Use connected components labeling to fill holes (except those connected to the image border)
        _, cc, stats, _ = cv.connectedComponentsWithStats(cv.bitwise_not(mask))
        h, w = image.shape
        index_small = np.where(
            (stats[:,cv.CC_STAT_LEFT] > 0) &
            (stats[:,cv.CC_STAT_LEFT] + stats[:,cv.CC_STAT_WIDTH] < w-1) &
            (stats[:,cv.CC_STAT_TOP] > 0) &
            (stats[:,cv.CC_STAT_TOP] + stats[:,cv.CC_STAT_HEIGHT] < h-1)
            )
        mask[np.isin(cc, index_small)] = 255
        if intermediate_results is not None: intermediate_results.append((np.copy(mask), 'After fill holes'))
        
        if parameters.opening_count > 0:
            # Applies opening to remove small blobs and protrusions
            mask = cv.morphologyEx(mask, cv.MORPH_OPEN, self._se3x3, iterations = parameters.opening_count)    
            if intermediate_results is not None: intermediate_results.append((np.copy(mask), 'After opening'))
        
        # The previous step may have created more cc: keep only the largest
        self._remove_other_cc(mask)
        
        return mask


    def _remove_other_cc(self, mask):
        num, cc, stats, _ = cv.connectedComponentsWithStats(mask)
        if num > 1:
            index = np.argmax(stats[1:,cv.CC_STAT_AREA]) + 1
            mask[cc!=index] = 0        


    _se3x3 = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
