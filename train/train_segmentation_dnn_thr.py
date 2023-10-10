##### Script to optimize the threshold parameter of the DnnSegmentationAlgorithm on FVC datasets B #####

import numpy as np
import cv2 as cv
import tensorflow as tf

import sys
sys.path.append(".") # To import packages from this project
import pyfing as pf
from pyfing.segmentation import compute_segmentation_error
from common.fvc_segmentation_utils import fvc_db_non_500_dpi, load_db, load_gt


PATH_FVC = '../datasets/'
PATH_GT = '../datasets/segmentationbenchmark/groundtruth/'
PATH_RES = '../results/'

##

print("Threshold optimization...")
alg = pf.DnnSegmentationAlgorithm()
TRAIN_DATASETS = [(y, db, "b") for y in (2000, 2002, 2004) for db in (1,2,3,4)]
for y, db, subset in TRAIN_DATASETS:
    images = load_db(PATH_FVC, y, db, subset)
    gt = load_gt(PATH_GT, y, db, subset)
    best_err = 100
    best_t = 0.5
    alg.parameters.image_dpi = fvc_db_non_500_dpi.get((y, db), 500)
    for index in range(21):
        alg.parameters.threshold = index * 0.05
        masks = alg.run_on_db(images)
        err = sum(compute_segmentation_error(m, x) for m, x in zip(masks, gt)) / len(images)
        if err < best_err:
            best_t = alg.parameters.threshold
            best_err = err
        print(f"{y} {db}: t = {alg.parameters.threshold} -> {err} (best = {best_err} @ {best_t})")
    alg.parameters.threshold = best_t
    alg.parameters.save(PATH_RES + f"fvc{y}_db{db}_b_dnn_params.json")

