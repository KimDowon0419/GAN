import numpy as np
import test0713
import cv2
import base64
from PIL import Image
import os , io , sys
import json
from flask import Flask, request,render_template,Response,jsonify,flash,redirect, url_for, abort
from werkzeug.utils import secure_filename
from functools import wraps
import matplotlib.pyplot as plt


import argparse


import tensorflow as tf
import neuralgym as ng
import scipy.misc
import glob
import random
import time
from tensorflow.python.platform import gfile

from scipy import signal


import progressbar

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

# Root directory of the project
ROOT_DIR = os.path.abspath("C:/Users/HACTSLAB/PycharmProjects/hifill_for_test/hifill_maskrcnn")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
# Import COCO config
sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version
from samples.coco import coco
import glob
import timeit

app = Flask(__name__)

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

parser = argparse.ArgumentParser()

#parser.add_argument('--image_dir', default='./data/test/testset_resize/', type=str,
#                    help='The directory of images to be completed.')
#parser.add_argument('--mask_dir', default='./data/test/maskset_resize/', type=str,
#                    help='The directory of masks, value 255 indicates mask.')
parser.add_argument('--image_dir', default='C:/Users/HACTSLAB/PycharmProjects/hifill_for_test/hifill_maskrcnn/samples/static/image/', type=str,
                    help='The directory of images to be completed.')
parser.add_argument('--mask_dir', default='C:/Users/HACTSLAB/PycharmProjects/hifill_for_test/hifill_maskrcnn/samples/maskset/', type=str,
                    help='The directory of masks, value 255 indicates mask.')

parser.add_argument('--output_dir', default='C:/Users/HACTSLAB/PycharmProjects/hifill_for_test/hifill_maskrcnn/samples/output/', type=str,
                    help='Where to write output.')

#parser.add_argument('--checkpoint_dir', default='./model_logs/20210330132201_model_HD_512/', type=str,
#                    help='The directory of tensorflow checkpoint.')
parser.add_argument('--checkpoint_dir', default='C:/Users/HACTSLAB/PycharmProjects/hifill_for_test/hifill_maskrcnn/samples//20210330132201_model_HD_512/', type=str,
                    help='The directory of tensorflow checkpoint.')

parser.add_argument('--rectangle_mask', default=True, type=bool,
                    help='whether to use rectangle masks.')
parser.add_argument('--input_size', default=512, type=int,
                    help='The size of input image.')
parser.add_argument('--times', default=1, type=int,
                    help='The size of input image.')



args = parser.parse_args()


def sort(str_lst):
    return [s for s in sorted(str_lst)]

def read_imgs_masks(args):
    paths_img = glob.glob(args.image_dir+'/*.*[g|G]')
    paths_img = sort(paths_img)
    paths_mask = glob.glob(args.mask_dir+'/*.*[g|G]')
    paths_mask = sort(paths_mask)
    return paths_img, paths_mask

def get_input(path_img, path_mask):
    image = cv2.imread(path_img)
    mask = cv2.imread(path_mask)
    mask = cv2.bitwise_not(mask)

    image = np.expand_dims(image, 0)
    mask = np.expand_dims(mask, 0)
    return np.concatenate([image, mask], axis=2), image[0], mask[0]


if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)


@app.route('/inference', methods=['POST'])
def inference():
    o_data = request.json
    data = o_data['img']
    dir = o_data['dir']
    data = base64.b64decode(data)
    jpg_arr = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(jpg_arr, cv2.IMREAD_COLOR)


    # 영상 정보 불러오기
    # video = cv2.VideoCapture('./201106.mp4')

    # Directory to save logs and trained model
    MODEL_DIR = os.path.join(ROOT_DIR, "logs")

    # Local path to trained weights file
    COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
    # Download COCO trained weights from Releases if needed
    if not os.path.exists(COCO_MODEL_PATH):
        utils.download_trained_weights(COCO_MODEL_PATH)

    # Directory of images to run detection on
    # IMAGE_DIR = os.path.join(ROOT_DIR, "samples/static/image/")

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

    # image_dir_path = glob.glob(IMAGE_DIR+'*.*[gG]')

    # Load a random image from the images folder
    # for path_img in image_dir_path:
    # image = skimage.io.imread(path_img)

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
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, hspace=0, wspace=0)
    plt.savefig(fname=MASK_DIR + '/' + os.path.basename(dir), bbbox_inches='tight', pad_inches=0)
    # plt.show()

    # running time
    print("Occlusion removal running time: ", Occlusion_removal_running_time)

    Total_running_time = time.time() - start_time
    print("Total running time: ", Total_running_time)


    paths_img, paths_mask = read_imgs_masks(args)


    with tf.Graph().as_default():
        with tf.gfile.FastGFile('hifill_0407.pb', "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name="")
        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            image_ph = sess.graph.get_tensor_by_name('img:0')
            mask_ph = sess.graph.get_tensor_by_name('mask:0')
            outputs = sess.graph.get_tensor_by_name('outputs:0')

            total_time = 0.

            bar = progressbar.ProgressBar(maxval=len(paths_img),
                                          widgets=[progressbar.Bar('=', '[', ']'), ' ',
                                                   progressbar.Percentage()])
            bar.start()
            for (i, path_img) in enumerate(paths_img):
                rint = i % len(paths_mask)
                bar.update(i + 1)
                in_img, img, mask = get_input(path_img, paths_mask[rint])
                s = time.time()

                outputs_arr = sess.run(outputs, feed_dict={image_ph: img, mask_ph: 255 - mask})
                res = outputs_arr[0]
                total_time += time.time() - s
                img_hole = img * (1 - mask / 255) + mask
                res = np.concatenate([img, img_hole, res], axis=1)
                cv2.imwrite(args.output_dir + '/' + str(i) + '.jpg', res)
            bar.finish()
            print('average time per image', total_time / len(paths_img))

    inpt = res
    result2 = inpt.tolist()
    dowon = {'inpt': result2}
    return jsonify(dowon)



@app.route("/")
def upload_form():
     views = os.listdir('static/resize/')
     views = ['resize/' + file for file in views]
     return render_template("test.html", views=views)



if __name__ == '__main__':
    app.debug = True
    app.run(host='0.0.0.0', port=2431, threaded=False)
