import math
import numpy as np
import cv2 as cv
from .definitions import Image
from .segmentation import SegmentationParameters, SegmentationAlgorithm


class GradMagSegmentationParameters(SegmentationParameters):
    """
    Parameters of the GradMag segmentation algorithm.    
    """

    def __init__(self, sigma = 13/3, percentile = 95, threshold = 0.2, closing_count = 6, opening_count = 12, image_dpi = 500):
        """
        TODO ...
        """
        self.sigma = sigma
        self.percentile = percentile
        self.threshold = threshold
        self.closing_count = closing_count
        self.opening_count = opening_count
        self.image_dpi = image_dpi


class GradMagSegmentationAlgorithm(SegmentationAlgorithm):
    """
    TODO
    """

    def __init__(self, parameters : GradMagSegmentationParameters = GradMagSegmentationParameters()):
        super().__init__(parameters)
        self.parameters = parameters

    def run(self, image: Image, intermediate_results = None) -> Image:
        parameters = self.parameters

        # Resizes the image if its resolution is not 500 dpi
        image_h, image_w = image.shape
        if parameters.image_dpi != 500:
            f = 500 / parameters.image_dpi
            image = cv.resize(image, None, fx = f, fy = f, interpolation = cv.INTER_CUBIC)

        # Calculates the gradient magnitude
        gx, gy = cv.spatialGradient(image)
        sm = cv.magnitude(gx.astype(np.float32), gy.astype(np.float32))
        if intermediate_results is not None:
            intermediate_results.append((sm, 'Gradient magnitude'))

        # Averages the gradient magnitude with a Gaussian filter
        gs =  math.ceil(3 * parameters.sigma) * 2 + 1
        r = cv.GaussianBlur(sm, (gs, gs), parameters.sigma)
        if intermediate_results is not None:
            intermediate_results.append((r, 'Average gradient magnitude'))

        # Compute the actual threshold
        norm_t = np.percentile(sm, parameters.percentile) * parameters.threshold

        # Selects pixels with average gradient magnitude above the threshold
        mask = cv.threshold(r, norm_t, 255, cv.THRESH_BINARY)[1].astype(np.uint8)
        if intermediate_results is not None:
            intermediate_results.append((np.copy(mask), 'Thresholding'))

        if parameters.closing_count > 0:
            # Applies closing to fill small holes and concavities
            mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, self._se3x3, iterations = parameters.closing_count)
            if intermediate_results is not None:
                intermediate_results.append((np.copy(mask), 'After closing'))

        # Remove all but the largest component
        self._remove_other_cc(mask)
        if intermediate_results is not None:
            intermediate_results.append((np.copy(mask), 'Largest component'))

        # Use connected components labeling to fill holes (except those connected to the image border)
        _, cc, stats, _ = cv.connectedComponentsWithStats(cv.bitwise_not(mask)) # in the background
        h, w = image.shape
        holes = np.where((stats[:,cv.CC_STAT_LEFT] > 0) &
                         (stats[:,cv.CC_STAT_LEFT] + stats[:,cv.CC_STAT_WIDTH] < w-1) &
                         (stats[:,cv.CC_STAT_TOP] > 0) &
                         (stats[:,cv.CC_STAT_TOP] + stats[:,cv.CC_STAT_HEIGHT] < h-1))
        mask[np.isin(cc, holes)] = 255
        if intermediate_results is not None:
            intermediate_results.append((np.copy(mask), 'After fill holes'))

        if parameters.opening_count > 0:
            # Applies opening to remove small blobs and protrusions
            mask = cv.morphologyEx(mask, cv.MORPH_OPEN, self._se3x3, iterations = parameters.opening_count)
            if intermediate_results is not None:
                intermediate_results.append((np.copy(mask), 'After opening'))

        # The previous step may have created more cc: keep only the largest
        self._remove_other_cc(mask)

        if parameters.image_dpi != 500:
            mask = cv.resize(mask, (image_w, image_h), interpolation = cv.INTER_NEAREST)

        return mask


    def _remove_other_cc(self, mask):
        num, cc, stats, _ = cv.connectedComponentsWithStats(mask)
        if num > 1:
            index = np.argmax(stats[1:,cv.CC_STAT_AREA]) + 1
            mask[cc!=index] = 0


    _se3x3 = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
