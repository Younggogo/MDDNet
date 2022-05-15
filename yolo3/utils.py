"""Miscellaneous utility functions."""

from functools import reduce
import tifffile as tr
from PIL import Image, ImageDraw
import numpy as np
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
import cv2
import matplotlib.pylab as plt
import re

def compose(*funcs):
    """Compose arbitrarily many functions, evaluated left to right.

    Reference: https://mathieularose.com/function-composition-in-python/
    """
    # return lambda x: reduce(lambda v, f: f(v), funcs, x)
    if funcs:
        return reduce(lambda f, g: lambda *a, **kw: g(f(*a, **kw)), funcs)
    else:
        raise ValueError('Composition of empty sequence not supported.')

def letterbox_image(image, size):
    '''resize image with unchanged aspect ratio using padding'''
    # iw, ih = image.size
    ih, iw, ic = image.shape
    w, h = size
    scale = min(w/iw, h/ih)
    nw = int(iw*scale)
    nh = int(ih*scale)
    dx = int(rand(0, w-nw))
    dy = int(rand(0, h-nh))
    image2 = np.zeros((h, w, ic))
    for i in range(ic):
        img = Image.fromarray(image[:, :, i])
        img = img.resize((nw,nh), Image.BICUBIC)
        new_image = Image.new('I', (w, h), 128)
        new_image.paste(img, (dx, dy))
        # new_image = np.array(new_image, dtype='float32')
        image2[:,:,i] = np.array(new_image)
    return image2

def rand(a=0, b=1):
    return np.random.rand()*(b-a) + a

def get_random_data(annotation_line, input_shape, random=True, max_boxes=20, jitter=.3, hue=.1, sat=1.5, val=1.5, proc_img=True):
    '''random preprocessing for real-time data augmentation'''
    line = annotation_line.split()
    image = tr.imread(line[0])
    ic, ih, iw = image.shape  #
    image = np.transpose(image,(1,2,0)) # 相当于转置 (216,409,25)
    image = normalgray(image)
    h, w = input_shape
    # box
    box = np.array([np.array(list(map(int,box.split(',')))) for box in line[1:]])


    if not random:
        scale = min(w/iw, h/ih)
        nw = int(iw*scale)
        nh = int(ih*scale)
        dx = (w-nw)//2
        dy = (h-nh)//2
        image_data=np.zeros((h, w, ic))
        if proc_img:
            for i in range(ic):
                img = Image.fromarray(image[:, :, i])
                img = img.resize((nw, nh), Image.BICUBIC)
                new_image = Image.new('I', (w, h), 128)
                new_image.paste(img, (dx, dy))
                image_data[:, :, i] = np.array(new_image)
            image_data /=255.

        # correct boxes
        box_data = np.zeros((max_boxes,5))
        if len(box)>0:
            np.random.shuffle(box)
            if len(box)>max_boxes: box = box[:max_boxes]
            box[:, [0,2]] = box[:, [0,2]]*scale + dx
            box[:, [1,3]] = box[:, [1,3]]*scale + dy
            box_data[:len(box)] = box

        return image_data, box_data

    # resize image
    new_ar = w/h * rand(1-jitter,1+jitter)/rand(1-jitter,1+jitter)
    scale = rand(.25, 2)
    if new_ar < 1:
        nh = int(scale*h)
        nw = int(nh*new_ar)
    else:
        nw = int(scale*w)
        nh = int(nw/new_ar)

    # place image
    dx = int(rand(0, w-nw))
    dy = int(rand(0, h-nh))
    imagere = np.zeros((h, w, ic))
    for i in range(ic):
        imgre = Image.fromarray(image[:, :, i])
        imgre = imgre.resize((nw, nh), Image.BICUBIC)  # 将图像直接resize
        new_image = Image.new('I', (w, h), 128)
        new_image.paste(imgre, (dx, dy))  # 将img贴在new-image上（dx,dy）部分
        # new_image = np.array(new_image, dtype='float32')
        imagere[:, :, i] = np.array(new_image)


    # flip image or not
    image = imagere
    flip = rand()<.5
    #if flip: image = image.transpose(Image.FLIP_LEFT_RIGHT)         
    if flip:
        imageflip = np.zeros((h, w, ic))
        for i in range (ic):                                   #   改

            imgflip = Image.fromarray(image[:, :, i])
            imgflip = imgflip.transpose(Image.FLIP_LEFT_RIGHT) # on flip
            imgflip = np.array(imgflip, dtype='float32')
            imageflip[:, :, i] = imgflip
        image = imageflip
    image_data = np.array(image)/255.

    ''' 加噪声，在使用matlab进行原始图像处理时，每一幅图被加了
    # distort image                   #   扭曲图像           #  不太好做啊  不能用单通道
    hue = rand(-hue, hue)
    sat = rand(1, sat) if rand()<.5 else 1/rand(1, sat)
    val = rand(1, val) if rand()<.5 else 1/rand(1, val)
    x = rgb_to_hsv(np.array(image)/255.)
    x[..., 0] += hue
    x[..., 0][x[..., 0]>1] -= 1
    x[..., 0][x[..., 0]<0] += 1
    x[..., 1] *= sat
    x[..., 2] *= val
    x[x>1] = 1
    x[x<0] = 0
    image_data = hsv_to_rgb(x) # numpy array, 0 to 1
    '''

    box_data = np.zeros((max_boxes,5))
    if len(box)>0:
        np.random.shuffle(box)
        box[:, [0,2]] = box[:, [0,2]]*nw/iw + dx
        box[:, [1,3]] = box[:, [1,3]]*nh/ih + dy
        if flip: box[:, [0,2]] = w - box[:, [2,0]]
        box[:, 0:2][box[:, 0:2]<0] = 0
        box[:, 2][box[:, 2]>w] = w
        box[:, 3][box[:, 3]>h] = h
        box_w = box[:, 2] - box[:, 0]
        box_h = box[:, 3] - box[:, 1]
        box = box[np.logical_and(box_w>1, box_h>1)] # discard invalid box
        if len(box)>max_boxes: box = box[:max_boxes]
        box_data[:len(box)] = box
    '''
    print(box_data[0][0], box_data[0][1], box_data[0][2], box_data[0][3])     #  读取图片、框后，检查是否正确
    img = Image.fromarray((image_data[:,:,10:13]*255).astype(np.uint8))
    draw = ImageDraw.Draw(img)
    draw.rectangle([box_data[0][0], box_data[0][1], box_data[0][2], box_data[0][3]],outline=255)
    img.show()
    '''
    return image_data, box_data

def noise(img,snr):
    h=img.shape[0]
    w=img.shape[1]
    img1=img.copy()
    sp=h*w
    NP=int(sp*(1-snr))
    for i in range (NP):
        randx=np.random.randint(1,h-1)
        randy=np.random.randint(1,w-1)
        if np.random.random()<=0.5:
            img1[randx,randy]=0
        else:
            img1[randx,randy]=255
    return img1



def normalgray(image):
    ih, iw, ic = image.shape
    msnormal = np.zeros((ih, iw, ic)).astype(np.uint8)
    for i in range(ic):
        midimage = image[:,:,i]
        msmin = midimage.min()
        msmax = midimage.max()
        msnormal[:,:,i] = (((midimage-msmin) / (msmax-msmin))*255).astype(np.uint8)
    return msnormal

def ms_imshow(image):
    ih, iw, ic = image.shape
    images = np.zeros((ih, iw, 3))
    # images = images.astype(np.int)
    images[:, :, 0] = image[:, :, 5]
    images[:, :, 1] = image[:, :, 7]
    images[:, :, 2] = image[:, :, 8]
    return images

def predicted_image(scores, boxes, dc, save_name=None):
    path = r'D:\tiny-yolov3-potatoes-fpg\VOC2007\Detectionresult\\'
    # file_path = path + re.sub("\D", "", save_name) + '.txt'
    file_path = path + save_name[0:7] + '.txt'
    file_handle = open(file_path, mode='w')
    LABEL = ["bug_eye", "dry_rot", "common_scab", "bruise", "germination"]
    index = dc[0].astype(int)
    for i in range(len(scores)):
            # onestr = np.row_stack((LABEL[1],np.stack((scores[i,1],boxes[i,:]),axis=1)))
        pv = str(float('%.6f' % scores[i]))
        xn = str(boxes[i, 1].astype(int))
        yn = str(boxes[i, 0].astype(int))
        xx = str(boxes[i, 3].astype(int))
        yx = str(boxes[i, 2].astype(int))
            # boxindex = np.hstack((float('%.6f' % scores[i, 0]), str(boxes[i,:].astype(int))))
            # onestr = str(np.hstack((LABEL[0], str(boxindex))))
            # onestr1 = onestr.replace('[','').replace(']','').replace("'","").replace('"','')
            # file_handle.writelines(onestr1)
        file_handle.write(LABEL[index] + ' ' + pv + ' ' + xn + ' ' + yn + ' ' + xx + ' ' + yx + ' ')
        file_handle.writelines('\n')
            # boxindex = np.stack((scores[0,1],boxes),axis=1)
        # namebox = np.row_stack((LABEL[1],boxindex))
    file_handle.close()