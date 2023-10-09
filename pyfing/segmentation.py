from abc import abstractmethod, ABC
import numpy as np
import cv2 as cv
from .definitions import Image, Parameters


class SegmentationParameters(Parameters):
    """
    Base class for segmentation algorithm parameters.
    """
    pass


class SegmentationAlgorithm(ABC):
    """
    Base class for segmentation algorithms.
    """
    def __init__(self, parameters: SegmentationParameters):
        self.parameters = parameters
    
    @abstractmethod
    def run(self, image: Image, intermediate_results = None) -> Image:
        raise NotImplementedError
    
    def run_on_db(self, images: [Image]) -> [Image]:
        return [self.run(img) for img in images]


def compute_segmentation_error(mask, gt_mask):
    """Returns the segmentation error (percentage of wrong pixels) of mask with respect to ground truth mask gt_mask"""
    return np.count_nonzero(gt_mask - mask) / mask.size * 100


def compute_dice_coefficient(mask, gt_mask):
    """Returns the Dice Coefficient of mask with respect to ground truth mask gt_mask"""
    return 2 * np.count_nonzero(gt_mask*mask) / (np.count_nonzero(gt_mask) + np.count_nonzero(mask))

