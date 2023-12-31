import numpy as np
import cv2 as cv
import tensorflow as tf
from .definitions import Image
from .segmentation import SegmentationParameters, SegmentationAlgorithm


class DnnSegmentationParameters(SegmentationParameters):  
    """
    Parameters of the DNN segmentation algorithm.    
    """
    
    def __init__(self, model_name = "segmentation_512.keras", dnn_input_dpi = 500, dnn_input_size = (512, 512), image_dpi = 500, threshold = 0.5):
        self.model_name = model_name        
        self.dnn_input_size = dnn_input_size
        self.dnn_input_dpi = dnn_input_dpi
        self.image_dpi = image_dpi
        self.threshold = threshold


class DnnSegmentationAlgorithm(SegmentationAlgorithm):
    """
    TODO
    """

    def __init__(self, parameters = DnnSegmentationParameters(), models_folder = "./models/"):
        super().__init__(parameters)
        self.parameters = parameters
        self.models_folder = models_folder
        if self.parameters.model_name != "":
            self.load_model()

    def load_model(self):
        """
        Loads the keras model parameters.model_name.
        """
        self.model = tf.keras.models.load_model(self.models_folder + self.parameters.model_name)

    def run(self, image: Image, intermediate_results = None) -> Image:
        h, w = image.shape
        ir = self._adjust_input(image)
        if intermediate_results is not None: intermediate_results.append((np.copy(ir), 'Adjusted input'))
        val_preds = self.model.predict(ir[np.newaxis,...,np.newaxis],verbose = 0)
        if intermediate_results is not None: intermediate_results.append((np.copy(val_preds[0].squeeze()), 'Net output'))
        mask = np.where(val_preds[0].squeeze()<self.parameters.threshold, 0, 255).astype(np.uint8) # [0..1] ==> 0,255
        return self._adjust_output(mask, (w, h))

    def run_on_db(self, images: [Image]) -> [Image]:
        ar = np.array([self._adjust_input(image)[...,np.newaxis] for image in images])
        masks = np.where(self.model.predict(ar, verbose = 0)<self.parameters.threshold,0,255).astype(np.uint8) # [0..1] ==> 0,255
        return [self._adjust_output(mask.squeeze(), image.shape[::-1]) for mask,image in zip(masks, images)]

    def _adjust_input(self, image):
        if self.parameters.dnn_input_dpi != self.parameters.image_dpi:
            # Resize to make its resolution parameters.dnn_input_dpi
            f = self.parameters.dnn_input_dpi / self.parameters.image_dpi
            image = cv.resize(image, None, fx = f, fy = f, interpolation = cv.INTER_CUBIC)
        image = (255 - image).astype(np.uint8) # The network is trained on negative images
        return _adjust_size(image, self.parameters.dnn_input_size, 0) 

    def _adjust_output(self, mask, size):
        if self.parameters.dnn_input_dpi != self.parameters.image_dpi:
            # Resize to target resolution
            f = self.parameters.image_dpi / self.parameters.dnn_input_dpi
            mask = cv.resize(mask, None, fx = f, fy = f, interpolation = cv.INTER_NEAREST)
        # Crop or add borders
        return _adjust_size(mask, size, 0)


def _adjust_size(image, target_size, border_value):
    # For each side computes crop size (if negative) or border to be added (if positive)
    h, w = image.shape
    target_w, target_h = target_size
    left = (target_w - w) // 2
    right = target_w - w - left
    top = (target_h - h) // 2
    bottom = target_h - h - top
    if left < 0 or right < 0: # Horizontal crop
        image = image[:, -left:(right if right < 0 else w)]
    if top < 0 or bottom < 0: # Vertical crop
        image = image[-top:(bottom if bottom < 0 else h)]    
    if left > 0 or right > 0 or top > 0 or bottom > 0: # Add borders
        image = cv.copyMakeBorder(image, max(0,top), max(0,bottom), max(0,left), max(0,right), cv.BORDER_CONSTANT, value = border_value)
    return image

