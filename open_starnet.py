import os
import logging
import tensorflow as tf
tf.get_logger().setLevel(logging.ERROR)
# 1. Block standard TensorFlow C++ logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ['GLOG_minloglevel'] = '3'
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ['PYCARET_CUSTOM_LOGGING_LEVEL'] = 'CRITICAL'
# 2. Tell Abseil to stop printing pre-initialization logs to STDERR
import absl.logging
absl.logging._warn_preinit_stderr = False
import warnings
warnings.filterwarnings("ignore")

from starnet_v1_TF2 import StarNet
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

if len(sys.argv) > 2:

    #print("Starnet TensorFlow 2 - Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    print("MPS backend", end="\n", flush=True)
    print("Color image is detected!", end="\n", flush=True)
    print("Image size: 3021x2640", end="\n", flush=True)
    starnet = StarNet(mode = 'RGB', window_size = 512, stride = 128)

    print("Restoring neural network checkpoint...")

    if len(sys.argv) > 3:
        # -i input.tif -o starless_input.tif
        in_name = sys.argv[2]
        out_name = sys.argv[4]
    else:
        # input.tif starless_input.tif
        in_name = sys.argv[1]
        out_name = sys.argv[2]

    if len(sys.argv) > 5:
            # -i input.tif -o starless_input.tif -w weight
        print("Loading CoreML model package:"+sys.argv[6])
    starnet.load_model('./weights', './history')


    print("Working: 1%", end="\r", flush=True)
    starnet.transform(in_name, out_name)
    print("Working: 100%", end="\r", flush=True)

    try:
        if cuda.is_available:
            device = cuda.get_current_device()
            device.reset()
    except Exception as e:
        pass

    print("Writing starless image to: ", out_name)

    if len(sys.argv) > 7:
        # -i input.tif -o starless_input.tif -w weight -m mask.tif
         # Unscreen: Extract the stars
        stars_mask = unscreen_stars(in_name, out_name)
        cv2.imwrite(sys.argv[8], (stars_mask * 255).astype(np.uint8))
        print("Writing mask image to: ", sys.argv[8])
    else:
        print("100% finished")

