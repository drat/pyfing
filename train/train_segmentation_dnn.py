##### Script to train a model for the DnnSegmentationAlgorithm on FVC datasets B #####

import numpy as np
import cv2 as cv
import tensorflow as tf

import sys
sys.path.append(".") # To import packages from this project
import pyfing as pf
from pyfing.segmentation import compute_segmentation_error, compute_dice_coefficient
from common.fvc_segmentation_utils import fvc_db_non_500_dpi, load_db, load_gt


PATH_FVC = '../datasets/'
PATH_GT = '../datasets/segmentationbenchmark/groundtruth/'
PATH_RES = '../results/'


INPUT_IMAGE_SIZE = (512, 512)
INPUT_IMAGE_DPI = 500
FILTERS = [16, 32, 64, 128, 256, 512]
BUFFER_SIZE = 1000
TRAIN_STEPS = 3000
BATCH_SIZE = 16
MIN_EPOCHS = 51
MAX_EPOCHS = 500
PATIENCE = 21

AUGMENT_PROB = 0.5
AUGMENT_PROB_TRASLATION = 0.15
AUGMENT_PROB_ROTATION = 0.15
AUGMENT_PROB_CONTRAST = 0.15
AUGMENT_PROB_FLIP = 0.15
AUGMENT_PROB_ZOOM = 0.15
AUGMENT_MAX_TRASLATION = 0.2 #0.1
AUGMENT_MAX_ROTATION = 0.07 #0.05
AUGMENT_MAX_CONTRAST = 0.6
AUGMENT_MAX_ZOOM = 0.1


class FvcImageAugmentation(tf.keras.layers.Layer):
  def __init__(self, seed=42):
    super().__init__()
    l = tf.keras.layers    
    self.rotate_images = l.RandomRotation(AUGMENT_MAX_ROTATION, seed=seed, fill_mode="constant", fill_value=0)
    self.rotate_masks = l.RandomRotation(AUGMENT_MAX_ROTATION, seed=seed, interpolation="nearest", fill_mode="constant", fill_value=0)
    self.translate_images = l.RandomTranslation(AUGMENT_MAX_TRASLATION, AUGMENT_MAX_TRASLATION, seed=seed, fill_mode="constant", fill_value=0)
    self.translate_masks = l.RandomTranslation(AUGMENT_MAX_TRASLATION, AUGMENT_MAX_TRASLATION, seed=seed, interpolation="nearest", fill_mode="constant", fill_value=0)
    self.flip_images = l.RandomFlip("horizontal", seed=seed)
    self.flip_masks = l.RandomFlip("horizontal", seed=seed)
    self.contrast_images = l.RandomContrast((AUGMENT_MAX_CONTRAST, 0), seed=seed)
    self.zoom_images = l.RandomZoom(AUGMENT_MAX_ZOOM, seed=seed, fill_mode="constant", fill_value=0)
    self.zoom_masks = l.RandomZoom(AUGMENT_MAX_ZOOM, seed=seed, interpolation="nearest", fill_mode="constant", fill_value=0)

  def call(self, images, masks):
    if tf.random.uniform([]) < AUGMENT_PROB:
        masks = tf.cast(masks, tf.dtypes.float32)
        
        if tf.random.uniform([]) < AUGMENT_PROB_ROTATION:
            images = self.rotate_images(images)
            masks = self.rotate_masks(masks)
        if tf.random.uniform([]) < AUGMENT_PROB_TRASLATION:
            images = self.translate_images(images)
            masks = self.translate_masks(masks)
        if tf.random.uniform([]) < AUGMENT_PROB_FLIP:
            images = self.flip_images(images)
            images = tf.zeros_like(images)
        if tf.random.uniform([]) < AUGMENT_PROB_CONTRAST:
            images = self.contrast_images(images)
        if tf.random.uniform([]) < AUGMENT_PROB_ZOOM:
            images = self.zoom_images(images)
            masks = self.zoom_masks(masks)
        
        masks = tf.cast(masks, tf.dtypes.uint8)

    return images, masks

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


def _adjust_input(image, dpi, is_mask = False):
    if dpi != INPUT_IMAGE_DPI:
        # Resize to make its resolution INPUT_IMAGE_DPI dpi
        f = INPUT_IMAGE_DPI / dpi
        image = cv.resize(image, None, fx = f, fy = f, interpolation = cv.INTER_NEAREST if is_mask else cv.INTER_CUBIC) # TODO provare vari tipi di interpolazione?
    if not is_mask:
        image = (255 - image).astype(np.uint8)
    return _adjust_size(image, INPUT_IMAGE_SIZE, 0) 


def load_fvc_dataset(db_years, db_numbers, db_set, impression_from, impression_to, max_fingers = None):
    i1, i2 = (1, 100) if db_set=="a" else (101, 110)
    if max_fingers is not None and i2-i1+1 > max_fingers:
        i2 = i1 + max_fingers - 1
    j1, j2 = impression_from, impression_to

    count = len(db_years) * len(db_numbers) * (i2-i1+1) * (j2-j1+1)
    x = np.empty((count, INPUT_IMAGE_SIZE[1], INPUT_IMAGE_SIZE[0], 1), dtype = np.float32)
    y = np.empty((count, INPUT_IMAGE_SIZE[1], INPUT_IMAGE_SIZE[0], 1), dtype = np.uint8)
    index = 0

    for year in db_years:
        for n in db_numbers:
            dpi = fvc_db_non_500_dpi.get((year, n), 500)
            for i in range(i1, i2+1):
                for j in range(j1, j2+1):
                    img_path = f'{PATH_FVC}fvc{year}/db{n}_{db_set}/{i}_{j}.png'
                    img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
                    if img is None:
                        raise Exception(f"Cannot load {img_path}")
                    img = _adjust_input(img, dpi)
                    gt_path = f'{PATH_GT}/fvc{year}_db{n}_im_{i}_{j}seg.png'
                    gt = cv.imread(gt_path, cv.IMREAD_GRAYSCALE)
                    gt = 255 - gt # FVC ground truth is 0 for foreground and 255 for background: we want 255 for foreground and 0 for background
                    gt = _adjust_input(gt, dpi, True) // 255 # 0,255 -> 0,1
                    if img is None:
                        raise Exception(f"Cannot load {gt_path}")
                    x[index] = img[..., np.newaxis]
                    y[index] = gt[..., np.newaxis]                        
                    index += 1
    return tf.data.Dataset.from_tensor_slices((x, y))

def get_model(levels):
    layers = tf.keras.layers
    img_size = INPUT_IMAGE_SIZE[::-1] # (w, h) ==> (h, w) = (rows, cols)
    inputs = layers.Input(img_size + (1,))
    x = inputs    
    level_outputs = []
    for filters in levels:
        x = layers.Conv2D(filters, 3, padding="same", activation = "relu")(x)
        x = layers.BatchNormalization()(x)
        level_outputs.append(x)
        x = layers.MaxPooling2D(2, padding="same")(x)
    for filters, lo in zip(reversed(levels), reversed(level_outputs)):
        x = layers.Conv2DTranspose(filters, 3, padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.UpSampling2D(2)(x)
        x = layers.Concatenate()([x, lo])
    outputs = layers.Conv2D(1, 3, activation="sigmoid", padding="same")(x)
    return tf.keras.Model(inputs, outputs)


def smooth_dice_coeff(y_true, y_pred):
    smooth = 1.
    y_true_f = tf.keras.backend.cast(tf.keras.backend.flatten(y_true), tf.dtypes.float32)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
    return score


def dice_loss(y_true, y_pred):
    loss = 1 - smooth_dice_coeff(y_true, y_pred)
    return loss


def train_model(model_name):
    print("Loading datasets...")
    train_ds = load_fvc_dataset([2000,2002,2004], [1,2,3,4], "b", 1, 7)
    train_ds = train_ds.cache().shuffle(BUFFER_SIZE).repeat().map(FvcImageAugmentation()).batch(BATCH_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE)
    val_ds = load_fvc_dataset([2000,2002,2004], [1,2,3,4], "b", 8, 8)
    val_ds = val_ds.batch(BATCH_SIZE)

    print("Preparing model...")
    model = get_model(FILTERS)
    model.summary()
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=[dice_loss])

    print("Training...")
    callbacks = [
        # TODO provare a fare monitor di val_loss in modo da togliere la metrica custom, oppure provare binary_accuracy
        tf.keras.callbacks.EarlyStopping(monitor='val_dice_loss', patience=PATIENCE, restore_best_weights=True, start_from_epoch=MIN_EPOCHS, min_delta=0.0002, verbose = 1), # 7, 5
    ]
    model.fit(train_ds, epochs=MAX_EPOCHS, steps_per_epoch=TRAIN_STEPS, validation_data=val_ds, callbacks=callbacks)

    print("Saving model...")
    model.save(PATH_RES+f'{model_name}.h5', include_optimizer=False)
    tf.keras.models.load_model(PATH_RES+f'{model_name}.h5').save(PATH_RES+f'{model_name}.keras')
    return model

def error_on_db(alg, images, gts, dpi):
    alg.parameters.image_dpi = dpi
    errors = [compute_segmentation_error(mask, gt) for mask, gt in zip(alg.run_on_db(images), gts)]
    return sum(errors) / len(errors)
##

print("Loading test dbs...")
TEST_DATASETS = [(y, db, "a") for y in (2000, 2002, 2004) for db in (1,2,3,4)]
test_dbs = [(load_db(PATH_FVC, year, db, subset), load_gt(PATH_GT, year, db, subset), fvc_db_non_500_dpi.get((year, db), 500)) for year, db, subset in TEST_DATASETS]

alg = pf.DnnSegmentationAlgorithm(pf.DnnSegmentationParameters(model_name=""), models_folder=PATH_RES)
for INPUT_IMAGE_SIZE, INPUT_IMAGE_DPI, FILTERS, model_name in [
    ((64, 64), 72, [16, 32, 64], "s_64"),
    ((128, 128), 125, [16, 32, 64, 128], "s_128"),
    ((256, 256), 250, [16, 32, 64, 128, 256], "s_256"),
    ((512, 512), 500, [16, 32, 64, 128, 256, 512], "s_512"),
    ((512, 512), 500, [16, 32, 64, 128, 256, 512], "s_512_1"),
    ((512, 512), 500, [16, 32, 64, 128, 256, 512], "s_512_2"),
    ((512, 512), 500, [16, 32, 64, 128, 256, 512], "s_512_3"),
    ((512, 512), 500, [16, 32, 64, 128, 256, 512], "s_512_4"),
    ((512, 512), 500, [16, 32, 64, 128, 256, 512], "s_512_5"),
    ((512, 512), 500, [16, 32, 64, 128, 256, 512], "s_512_6"),
    ((512, 512), 500, [16, 32, 64, 128, 256, 512], "s_512_7"),
    ((512, 512), 500, [16, 32, 64, 128, 256, 512], "s_512_8"),
    ((512, 512), 500, [16, 32, 64, 128, 256, 512], "s_512_9"),
    ((512, 512), 500, [16, 32, 64, 128, 256, 512], "s_512_10"),
    ((512, 512), 500, [16, 32, 64, 128, 256, 512], "s_512_11"),
    ((512, 512), 500, [16, 32, 64, 128, 256, 512], "s_512_12"),
]:    
    alg.model = train_model(model_name)
    alg.parameters.dnn_input_dpi = INPUT_IMAGE_DPI
    alg.parameters.dnn_input_size = INPUT_IMAGE_SIZE
    errors = [error_on_db(alg, images, gts, dpi) for images, gts, dpi in test_dbs]
    err = sum(errors) / len(errors)
    resultLine = f"{model_name} -> {err:.2f}%"
    with open(f'{PATH_RES}results.txt', 'a') as f:
        print(resultLine, file=f)                
    print(resultLine)
