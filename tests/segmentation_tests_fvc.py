##### Script to test both segmentation algorithms on FVC databases #####

import time
import json
import numpy as np
import cv2 as cv
import tensorflow as tf

import sys
sys.path.append(".") # To import packages from this project
import pyfing as pf
from pyfing.segmentation import compute_segmentation_error, compute_dice_coefficient
from common.fvc_segmentation_utils import load_db, load_gt

PATH_FVC = '../datasets/'
PATH_GT = '../datasets/segmentationbenchmark/groundtruth/'
PATH_PARAMS = './parameters/segmentation/'
PATH_RES = '../results/'

def compute_metrics(masks, gt):
    results = [(k / 8 + 1, (k % 8) + 1, (compute_segmentation_error(m, x), compute_dice_coefficient(m, x))) for k, (m, x) in enumerate(zip(masks, gt))]
    errors = np.array([e for _, _, (e, d) in results])
    dice_coeffs = np.array([d for _, _, (e, d) in results])
    return errors, dice_coeffs, results

def run_test(alg: pf.SegmentationAlgorithm, year, db, subset):
    # load dataset-specific parameters
    if isinstance(alg, pf.GradMagSegmentationAlgorithm):
        alg.parameters = pf.GradMagSegmentationParameters.load(f'{PATH_PARAMS}fvc{year}_db{db}_b_grad_mag_params.json') 
    else:
        alg.parameters = pf.DnnSegmentationParameters.load(f'{PATH_PARAMS}fvc{year}_db{db}_b_dnn_params.json')
        #alg.parameters = pf.DnnSegmentationParameters.load(f'{PATH_PARAMS}fvc{year}_db{db}_b_dnn_256_params.json')
        alg.parameters.threshold = 0.5
        #pass # TODO
    images = load_db(PATH_FVC, year, db, subset)
    gt = load_gt(PATH_GT, year, db, subset)
    start = time.time()
    masks = alg.run_on_db(images)
    elapsed = time.time() - start
    errors, dice_coeffs, results = compute_metrics(masks, gt)
    avg_err = np.mean(errors)
    avg_dice = np.mean(dice_coeffs)
    alg_name = type(alg).__name__
    print(f'Tested {alg_name} on FVC{year} DB{db}_{subset} ({len(images)} images). '\
          f'Err = {avg_err:5.2f}% [{errors.min():5.2f}%, {errors.max():5.2f}%] '\
          f'DC = {avg_dice:5.3f} [{dice_coeffs.min():5.3f}, {dice_coeffs.max():5.3f}] '\
          f'Tot time: {elapsed:5.2f}s Avg: {elapsed/len(images):.4f}s')    
    with open(f'{PATH_RES}fvc{year}_db{db}_{subset}_{alg_name}_res.txt', 'w') as file:
        json.dump(results, file)    
    return avg_err, avg_dice

## 

TEST_DATASETS = [(y, db, "a") for y in (2000, 2002, 2004) for db in (1,2,3,4)]

algs = [
    #pf.GradMagSegmentationAlgorithm(), 
    pf.DnnSegmentationAlgorithm(),
]

for alg in algs:
    print('Testing...')
    avg_err, avg_dices = np.mean(np.array([run_test(alg, y, db, subset) for y, db, subset in TEST_DATASETS]), 0)
    print(f'Avg: {avg_err:.2f}% {avg_dices:.3f}')
