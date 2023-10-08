import unittest
import numpy as np
import cv2 as cv
import tensorflow as tf

# TODO ???
import os, sys
sys.path.append(".")

import pyfing as pf


PATH_FVC = '../datasets/'
PATH_PARAMS = './parameters/segmentation/'
PATH_GT = '../datasets/segmentationbenchmark/groundtruth/'
PATH_TESTS = '../results/'


fingerprint = cv.imread(PATH_FVC + 'fvc2000/db2_b/101_1.png', cv.IMREAD_GRAYSCALE)
print(fingerprint.shape)
str = tf.config.list_physical_devices()
#seg = pf.GradMagSegmentationAlgorithm()
seg = pf.DnnSegmentationAlgorithm(tf.keras.models.load_model('./models/segmentation512x512.keras'))
mask = seg.run(fingerprint)
print(mask.shape)        

