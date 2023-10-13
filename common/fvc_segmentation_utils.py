import numpy as np
import cv2 as cv


fvc_db_non_500_dpi = {(2002,2): 569, (2004,3): 512} # FVC datasets with non-standard DPI

def load_db(path, year, db, subset):
    i1, i2 = (1, 100) if subset=="a" else (101, 110)
    j1, j2 = 1, 8
    return [cv.imread(f'{path}fvc{year}/db{db}_{subset}/{i}_{j}.png', cv.IMREAD_GRAYSCALE)
            for i in range(i1, i2+1) for j in range(j1, j2+1)]

def load_gt(path, year, db, subset):
    i1, i2 = (1, 100) if subset=="a" else (101, 110)
    j1, j2 = 1, 8
    return [cv.bitwise_not(cv.imread(f'{path}fvc{year}_db{db}_im_{i}_{j}seg.png', 
                                       cv.IMREAD_GRAYSCALE)) for i in range(i1, i2+1) for j in range(j1, j2+1)]
