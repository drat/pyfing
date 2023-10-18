##### Script to test both segmentation algorithms on FVC databases #####

import time
import json
import numpy as np

import sys
sys.path.append(".") # To import packages from this project
import pyfing as pf
from pyfing.segmentation import compute_segmentation_error, compute_dice_coefficient
from common.fvc_segmentation_utils import load_db, load_gt, fvc_db_non_500_dpi

PATH_FVC = '../datasets/'
PATH_GT = '../datasets/segmentationbenchmark/groundtruth/'
PATH_PARAMS = './parameters/segmentation/'
PATH_RES = '../results/'

def compute_metrics(masks, gt):
    results = [(k // 8 + 1, (k % 8) + 1, (compute_segmentation_error(m, x), compute_dice_coefficient(m, x))) for k, (m, x) in enumerate(zip(masks, gt))]
    errors = np.array([e for _, _, (e, d) in results])
    dice_coeffs = np.array([d for _, _, (e, d) in results])
    return errors, dice_coeffs, results

## 

year, db, subset = (2002, 4, "a")

alg = pf.GradMag2SegmentationAlgorithm()
print('Loading...')
alg.parameters = pf.GradMag2SegmentationParameters.load(f'{PATH_PARAMS}fvc{year}_db{db}_b_grad_mag2_params.json')
    #alg.parameters.threshold = 0.07
    #alg.parameters.closing_count = 1
    #alg.parameters.opening_count = 9
images = load_db(PATH_FVC, year, db, subset)
gt = load_gt(PATH_GT, year, db, subset)
for alg.parameters.sigma in np.arange(5.5, 6.7, 0.1):
#for alg.parameters.threshold in np.arange(0.03, 0.08, 0.01):
    #for alg.parameters.closing_count in range(0, 15):
        print(f'Testing s={alg.parameters.sigma} t={alg.parameters.threshold:.2f} cc={alg.parameters.closing_count} ...')    
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
        