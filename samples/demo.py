import os
import sys
import random
import math
import numpy as np
import skimage.io
import time
import matplotlib
import matplotlib.pyplot as plt
import cv2

import numpy as np
import tensorflow as tf
import argparse
import os
tf.config.list_physical_devices('GPU')

# Root directory of the project
ROOT_DIR = os.path.abspath("C:/Users/HACTSLAB/PycharmProjects/maskrcnn_hifill/hifill_maskrcnn/")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
# Import COCO config
sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version
from samples.coco import coco
import glob

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "samples/testset/")

class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
print(config.display())

# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)

# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']


image_dir_path = glob.glob(IMAGE_DIR+'*.*[gG]')

# Load a random image from the images folder
for path_img in image_dir_path:
    image = skimage.io.imread(path_img)

    # Run detection
    start_time = time.time()
    results = model.detect([image], verbose=1)
    Occlusion_removal_running_time = time.time() - start_time
    # Visualize results
    r = results[0]

    MASK_DIR = os.path.join(ROOT_DIR, "samples/maskset/")
    visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                                class_names, r['scores'])

    plt.axis('off'), plt.xticks([]), plt.yticks([])
    plt.tight_layout()
    plt.subplots_adjust(left = 0, bottom = 0, right = 1, top = 1, hspace = 0, wspace = 0)
    plt.savefig(fname = MASK_DIR + '/' + os.path.basename(path_img), bbbox_inches = 'tight', pad_inches=0)
    plt.show()

    # running time

    print("Occlusion removal running time: ", Occlusion_removal_running_time)




INPUT_SIZE = 512  # input image size for Generator
ATTENTION_SIZE = 32 # size of contextual attention

tf.compat.v1.disable_eager_execution()

def sort(str_lst):
    return [s for s in sorted(str_lst)]

# reconstruct residual from patches
def reconstruct_residual_from_patches(residual, multiple):
    residual = np.reshape(residual, [ATTENTION_SIZE, ATTENTION_SIZE, multiple, multiple, 3])
    residual = np.transpose(residual, [0,2,1,3,4])
    return np.reshape(residual, [ATTENTION_SIZE * multiple, ATTENTION_SIZE * multiple, 3])

# extract image patches
def extract_image_patches(img, multiple):
    h, w, c = img.shape
    img = np.reshape(img, [h//multiple, multiple, w//multiple, multiple, c])
    img = np.transpose(img, [0,2,1,3,4])
    return img

# residual aggregation module
def residual_aggregate(residual, attention, multiple):
    residual = extract_image_patches(residual, multiple * INPUT_SIZE//ATTENTION_SIZE)
    residual = np.reshape(residual, [1, residual.shape[0] * residual.shape[1], -1])
    residual = np.matmul(attention, residual)
    residual = reconstruct_residual_from_patches(residual, multiple * INPUT_SIZE//ATTENTION_SIZE)
    return residual

# resize image by averaging neighbors
def resize_ave(img, multiple):
    img = img.astype(np.float32)
    img_patches = extract_image_patches(img, multiple)
    img = np.mean(img_patches, axis=(2,3))
    return img

# pre-processing module
def pre_process(raw_img, raw_mask, multiple):

    raw_mask = raw_mask.astype(np.float32) / 255.
    raw_img = raw_img.astype(np.float32)

    # resize raw image & mask to desinated size
    large_img = cv2.resize(raw_img,  (multiple * INPUT_SIZE, multiple * INPUT_SIZE), interpolation = cv2. INTER_LINEAR)
    large_mask = cv2.resize(raw_mask, (multiple * INPUT_SIZE, multiple * INPUT_SIZE), interpolation = cv2.INTER_NEAREST)

    # down-sample large image & mask to 512x512
    small_img = resize_ave(large_img, multiple)
    small_mask = cv2.resize(raw_mask, (INPUT_SIZE, INPUT_SIZE), interpolation = cv2.INTER_NEAREST)

    # set hole region to 1. and backgroun to 0.
    small_mask = 1. - small_mask
    return large_img, large_mask, small_img, small_mask


# post-processing module
def post_process(raw_img, large_img, large_mask, res_512, img_512, mask_512, attention, multiple):

    # compute the raw residual map
    h, w, c = raw_img.shape
    low_base = cv2.resize(res_512.astype(np.float32), (INPUT_SIZE * multiple, INPUT_SIZE * multiple), interpolation = cv2.INTER_LINEAR)
    low_large = cv2.resize(img_512.astype(np.float32), (INPUT_SIZE * multiple, INPUT_SIZE * multiple), interpolation = cv2.INTER_LINEAR)
    residual = (large_img - low_large) * large_mask

    # reconstruct residual map using residual aggregation module
    residual = residual_aggregate(residual, attention, multiple)

    # compute large inpainted result
    res_large = low_base + residual
    res_large = np.clip(res_large, 0., 255.)

    # resize large inpainted result to raw size
    res_raw = cv2.resize(res_large, (w, h), interpolation = cv2.INTER_LINEAR)

    # paste the hole region to the original raw image
    mask = cv2.resize(mask_512.astype(np.float32), (w, h), interpolation = cv2.INTER_LINEAR)
    mask = np.expand_dims(mask, axis=2)
    res_raw = res_raw * mask + raw_img * (1. - mask)

    return res_raw.astype(np.uint8)


def inpaint(raw_img,
            raw_mask,
            sess,
            inpainted_512_node,
            attention_node,
            mask_512_node,
            img_512_ph,
            mask_512_ph,
            multiple):

    # pre-processing
    img_large, mask_large, img_512, mask_512 = pre_process(raw_img, raw_mask, multiple)

    # neural network
    inpaint_start = time.time()
    inpainted_512, attention, mask_512  = sess.run([inpainted_512_node, attention_node, mask_512_node], feed_dict={img_512_ph: [img_512] , mask_512_ph:[mask_512[:,:,0:1]]})
    Inpainting_running_time = time.time() - inpaint_start
    print("Inpainting running time: ", Inpainting_running_time)

    # post-processing
    res_raw_size = post_process(raw_img, img_large, mask_large, \
                 inpainted_512[0], img_512, mask_512[0], attention[0], multiple)

    return res_raw_size



def read_imgs_masks(args):
    paths_img = glob.glob(args.images+'/*.*[gG]')
    paths_mask = glob.glob(args.masks+'/*.*[gG]')
    paths_img = sort(paths_img)
    paths_mask = sort(paths_mask)
    print('#imgs: ' + str(len(paths_img)))
    print('#imgs: ' + str(len(paths_mask)))
    print(paths_img)
    print(paths_mask)
    return paths_img, paths_mask

parser = argparse.ArgumentParser()
args = parser.parse_args()
args.images = '../samples/testset' # input image directory
args.masks = '../samples/maskset' # input mask director
args.output_dir = './results' # output directory
args.multiple = 6 # multiples of image resizing

paths_img, paths_mask = read_imgs_masks(args)
if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)
with tf.Graph().as_default():
  with open('./pb/hifill.pb', "rb") as f:
    output_graph_def = tf.compat.v1.GraphDef()
    output_graph_def.ParseFromString(f.read())
    tf.import_graph_def(output_graph_def, name="")

  with tf.compat.v1.Session() as sess:
    #init = tf.global_variables_initializer()
    #sess.run(init)
    image_ph = sess.graph.get_tensor_by_name('img:0')
    mask_ph = sess.graph.get_tensor_by_name('mask:0')
    inpainted_512_node = sess.graph.get_tensor_by_name('inpainted:0')
    attention_node = sess.graph.get_tensor_by_name('attention:0')
    mask_512_node = sess.graph.get_tensor_by_name('mask_processed:0')

    for path_img, path_mask in zip(paths_img, paths_mask):
        raw_img = cv2.imread(path_img)
        raw_mask = cv2.imread(path_mask)
        inpainted = inpaint(raw_img, raw_mask, sess, inpainted_512_node, attention_node, mask_512_node, image_ph, mask_ph, args.multiple)
        filename = args.output_dir + '/' + os.path.basename(path_img)
        cv2.imwrite(filename + '_inpainted.jpg', inpainted)

# running time


Total_running_time = time.time() - start_time
print("Total running time: ", Total_running_time)
