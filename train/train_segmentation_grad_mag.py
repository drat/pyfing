##### Script find best parameters for GradMagSegmentationAlgorithm on FVC datasets B #####

import time
import json
import numpy as np
import cv2 as cv
import tensorflow as tf

import sys
sys.path.append(".") # To import the pyfing package from this project
import pyfing as pf
from pyfing.segmentation import compute_segmentation_error, compute_dice_coefficient


PATH_FVC = '../datasets/'
PATH_GT = '../datasets/segmentationbenchmark/groundtruth/'
PATH_RES = '../results/'


def load_db(year, db, subset):
    i1, i2 = (1, 100) if subset=="a" else (101, 110)
    j1, j2 = 1, 8
    return [cv.imread(f'{PATH_FVC}fvc{year}/db{db}_{subset}/{i}_{j}.png', cv.IMREAD_GRAYSCALE)
            for i in range(i1, i2+1) for j in range(j1, j2+1)]

def load_gt(year, db, subset):
    i1, i2 = (1, 100) if subset=="a" else (101, 110)
    j1, j2 = 1, 8
    return [cv.bitwise_not(cv.imread(f'{PATH_GT}/fvc{year}_db{db}_im_{i}_{j}seg.png', 
                                       cv.IMREAD_GRAYSCALE)) for i in range(i1, i2+1) for j in range(j1, j2+1)]

def average_err_on_db(images, gt, parameters):    
    alg = pf.GradMagSegmentationAlgorithm(parameters)
    masks = alg.run_on_db(images)
    errors = [compute_segmentation_error(m, x) for m, x in zip(masks, gt)]
    return np.mean(errors)

def optimize_parameters_in_ranges(year, db, subset, images, gt, best_parameters, ranges):    
    min_err = 100 if best_parameters is None else average_err_on_db(images, gt, best_parameters)
    for w in ranges[0]:
        for p in ranges[1]:  
            for t in ranges[2]:
                for cc in ranges[3]:
                    for oc in ranges[4]:
                        parameters = pf.GradMagSegmentationParameters(w, p, t, cc, oc)
                        e = average_err_on_db(images, gt, parameters)                
                        if e < min_err:
                            min_err = e
                            best_parameters = parameters
                            # saves best parameters to file
                            best_parameters.save(f'{PATH_RES}fvc{year}_db{db}_b_grad_mag_params.txt')
                        print(f'{year}-{db}-{subset}: {len(images)} fingerprints')
                        print(f'{parameters}\n-> {e:.2f}%\n(best: {best_parameters}\n-> {min_err:.2f}%)')
    return best_parameters

def optimize_parameters(year, db, subset):
    images = load_db(year, db, subset)
    gt = load_gt(year, db, subset)
    p = None
    try:
        p = pf.GradMagSegmentationParameters.load(f'{PATH_RES}fvc{year}_db{db}_b_grad_mag_params.txt')
        print(f'Loaded best parameters: {p}')
    except:
        print('Best parameters not found')
    p = optimize_parameters_in_ranges(year, db, subset, images, gt, p,
                                      [
                                          range(3,19,2),
                                          [85,90,95],
                                          [0.15,0.20,0.25,0.30],
                                          [7],
                                          [13],
                                      ])
    p = optimize_parameters_in_ranges(year, db, subset, images, gt, p,
                                      [
                                          [p.window_size],
                                          range(75,100,5),
                                          np.arange(0.05,0.4,0.01),
                                          [7],
                                          [13],
                                      ])
    p = optimize_parameters_in_ranges(year, db, subset, images, gt, p,
                                      [
                                          [p.window_size],
                                          [p.percentile],
                                          [p.threshold],
                                          range(0,15,3),
                                          range(0,24,3),
                                      ])

## 

TRAIN_DATASETS = [(y, db, "b") for y in (2000, 2002, 2004) for db in (1,2,3,4)]

for y, db, subset in TRAIN_DATASETS:
    optimize_parameters(y, db, subset)
