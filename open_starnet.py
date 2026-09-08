import tensorflow as tf
from PIL import Image as img
import logging
tf.get_logger().setLevel(logging.ERROR)
from starnet_v1_TF2 import StarNet
import tifffile as tiff
import sys
from numba import cuda
import cv2
import numpy as np

def unscreen_stars(original_path, starless_path):
    """
    Extracts a stars-only mask using the PixInsight PixelMath unscreen method.
    Formula: ~((~original) / (~starless))
    """
    # Load images in float32 format normalized between 0.0 and 1.0
    orig = cv2.imread(original_path).astype(np.float32) / 255.0
    starless = cv2.imread(starless_path).astype(np.float32) / 255.0

    # Invert both images: ~img is equivalent to (1.0 - img)
    inv_orig = 1.0 - orig
    inv_starless = 1.0 - starless

    # Prevent division by zero or negative values
    inv_starless = np.clip(inv_starless, 1e-7, 1.0)

    # Perform the division and invert the final result
    stars = 1.0 - (inv_orig / inv_starless)
    
    # Keep values bounded strictly within the [0, 1] range
    stars = np.clip(stars, 0.0, 1.0)
    
    return stars

if len(sys.argv) > 1:
    # -i input.tif -o starless_imput.tif
    print("Starnet TensorFlow 2 - Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    starnet = StarNet(mode = 'RGB', window_size = 512, stride = 128)
    starnet.load_model('./weights', './history')
    print("Weights Loaded!")
    in_name = sys.argv[2]
    out_name = sys.argv[4]
    starnet.transform(in_name, out_name)
    try:
        if cuda.is_available:
            device = cuda.get_current_device()
            device.reset()
    except Exception as e:
        print("Error resetting GPU: ", e)

    if len(sys.argv) > 4:
        # -i input.tif -o starless_imput.tif -w weight -m mask.tif
         # Unscreen: Extract the stars
        stars_mask = unscreen_stars(sys.argv[2], sys.argv[4])
        cv2.imwrite(sys.argv[8], (stars_mask * 255).astype(np.uint8))
    print("100% finished")

