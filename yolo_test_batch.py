# -*- coding: utf-8 -*-
"""
批量测试图像
测试的结果存放在路径outdir = "VOC2007/SegmentationClass
"""
import argparse
import os
from yolo import YOLO
from PIL import Image
import tifffile as tr
import numpy as np
from yolo3.utils import normalgray, ms_imshow
import matplotlib.pylab as plt
import glob
def detect_img(yolo):
    path = "VOC2007/Images/*.tif"
    outdir = "VOC2007/SegmentationClass"
    for jpgfile in glob.glob(path):
        jpgfileshow = jpgfile.replace('tif','jpg')
        imagejpg = Image.open(jpgfileshow)
        imagefile = jpgfile[15:24]
        image = tr.imread(jpgfile)
        ic, iw, ih = image.shape
        img = np.transpose(image, (1, 2, 0))
        img = normalgray(img)
        img = yolo.detect_image(img,imagejpg,imagefile)
        # img.show()
        img.save(os.path.join(outdir, os.path.basename(jpgfile)))
    yolo.close_session()
FLAGS = None

if __name__ == '__main__':
    # class YOLO defines the default value, so suppress any default here
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    '''
    Command line options
    '''
    parser.add_argument(
        '--model', type=str,
        help='path to model weight file, default ' + YOLO.get_defaults("model_path")
    )

    parser.add_argument(
        '--anchors', type=str,
        help='path to anchor definitions, default ' + YOLO.get_defaults("anchors_path")
    )

    parser.add_argument(
        '--classes', type=str,
        help='path to class definitions, default ' + YOLO.get_defaults("classes_path")
    )

    parser.add_argument(
        '--gpu_num', type=int,
        help='Number of GPU to use, default ' + str(YOLO.get_defaults("gpu_num"))
    )

    parser.add_argument(
        '--image', default=False, action="store_true",
        help='Image detection mode, will ignore all positional arguments'
    )
    '''
    Command line positional arguments -- for video detection mode
    '''
    parser.add_argument(
        "--input", nargs='?', type=str,required=False,default='./path2your_video',
        help = "Video input path"
    )

    parser.add_argument(
        "--output", nargs='?', type=str, default="",
        help = "[Optional] Video output path"
    )

    FLAGS = parser.parse_args()

    if FLAGS.image:
        """
        Image detection mode, disregard any remaining command line arguments
        """
        print("Image detection mode")
        if "input" in FLAGS:
             print("error")
    elif "input" in FLAGS:
         print("Image detection mode")
         print(" Ignoring remaining command line arguments: " + FLAGS.input + "," + FLAGS.output)
         detect_img(YOLO(**vars(FLAGS)))

