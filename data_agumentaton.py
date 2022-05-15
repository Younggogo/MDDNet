# -*- coding=utf-8 -*-
import os
import time
import random
import cv2
import numpy as np
from skimage import exposure
import xml.etree.ElementTree as ET
import xml.dom.minidom as DOC
import math
import tifffile as tr
# 从xml文件中提取bounding box信息, 格式为[[x_min, y_min, x_max, y_max, name]]
def parse_xml(xml_path):
    '''
    输入：
        xml_path: xml的文件路径
    输出：
        从xml文件中提取bounding box信息, 格式为[[x_min, y_min, x_max, y_max, name]]
    '''
    tree = ET.parse(xml_path)
    root = tree.getroot()
    objs = root.findall('object')
    coords = list()
    for ix, obj in enumerate(objs):
        name = obj.find('name').text
        box = obj.find('bndbox')
        x_min = int(box[0].text)
        y_min = int(box[1].text)
        x_max = int(box[2].text)
        y_max = int(box[3].text)
        coords.append([x_min, y_min, x_max, y_max, name])
    return coords

#将bounding box信息写入xml文件中, bouding box格式为[[x_min, y_min, x_max, y_max, name]]
def generate_xml(img_name,coords,img_size,out_root_path,cnt):
    '''
    输入：
        img_name：图片名称，如a.jpg
        coords:坐标list，格式为[[x_min, y_min, x_max, y_max, name]]，name为概况的标注
        img_size：图像的大小,格式为[h,w,c]
        out_root_path: xml文件输出的根路径
    '''
    doc = DOC.Document()  # 创建DOM文档对象

    annotation = doc.createElement('annotation')
    doc.appendChild(annotation)

    title = doc.createElement('folder')
    title_text = doc.createTextNode('Tianchi')
    title.appendChild(title_text)
    annotation.appendChild(title)

    title = doc.createElement('filename')
    title_text = doc.createTextNode(img_name)
    title.appendChild(title_text)
    annotation.appendChild(title)

    source = doc.createElement('source')
    annotation.appendChild(source)

    title = doc.createElement('database')
    title_text = doc.createTextNode('The Tianchi Database')
    title.appendChild(title_text)
    source.appendChild(title)

    title = doc.createElement('annotation')
    title_text = doc.createTextNode('Tianchi')
    title.appendChild(title_text)
    source.appendChild(title)

    size = doc.createElement('size')
    annotation.appendChild(size)

    title = doc.createElement('width')
    title_text = doc.createTextNode(str(img_size[1]))
    title.appendChild(title_text)
    size.appendChild(title)

    title = doc.createElement('height')
    title_text = doc.createTextNode(str(img_size[0]))
    title.appendChild(title_text)
    size.appendChild(title)

    title = doc.createElement('depth')
    title_text = doc.createTextNode(str(img_size[2]))
    title.appendChild(title_text)
    size.appendChild(title)

    for coord in coords:

        object = doc.createElement('object')
        annotation.appendChild(object)

        title = doc.createElement('name')
        title_text = doc.createTextNode(coord[4])
        title.appendChild(title_text)
        object.appendChild(title)

        pose = doc.createElement('pose')
        pose.appendChild(doc.createTextNode('Unspecified'))
        object.appendChild(pose)
        truncated = doc.createElement('truncated')
        truncated.appendChild(doc.createTextNode('1'))
        object.appendChild(truncated)
        difficult = doc.createElement('difficult')
        difficult.appendChild(doc.createTextNode('0'))
        object.appendChild(difficult)

        bndbox = doc.createElement('bndbox')
        object.appendChild(bndbox)
        title = doc.createElement('xmin')
        title_text = doc.createTextNode(str(int(float(coord[0]))))
        title.appendChild(title_text)
        bndbox.appendChild(title)
        title = doc.createElement('ymin')
        title_text = doc.createTextNode(str(int(float(coord[1]))))
        title.appendChild(title_text)
        bndbox.appendChild(title)
        title = doc.createElement('xmax')
        title_text = doc.createTextNode(str(int(float(coord[2]))))
        title.appendChild(title_text)
        bndbox.appendChild(title)
        title = doc.createElement('ymax')
        title_text = doc.createTextNode(str(int(float(coord[3]))))
        title.appendChild(title_text)
        bndbox.appendChild(title)

    # 将DOM对象doc写入文件
    f = open(os.path.join(out_root_path, str(cnt)+'.xml'),'w')
    f.write(doc.toprettyxml(indent = ''))
    f.close()

def show_pic(img, bboxes=None):
    '''
    输入:
        img:图像array
        bboxes:图像的所有boudning box list, 格式为[[x_min, y_min, x_max, y_max]....]
        names:每个box对应的名称
    '''
    cv2.imwrite('./1.jpg', img)
    img = cv2.imread('./1.jpg')
    for i in range(len(bboxes)):
        bbox = bboxes[i]
        x_min = bbox[0]
        y_min = bbox[1]
        x_max = bbox[2]
        y_max = bbox[3]
        cv2.rectangle(img,(int(x_min),int(y_min)),(int(x_max),int(y_max)),(0,255,0),3)
        cv2.putText(img, bbox[4], (int(x_min),int(y_min)), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255,255, 0), thickness=2)
    cv2.namedWindow('pic', 0)  # 1表示原图
    cv2.moveWindow('pic', 0, 0)
    cv2.resizeWindow('pic', 1200,800)  # 可视化的图片大小
    cv2.imshow('pic', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows() 
    os.remove('./1.jpg')

# 图像均为cv2读取
class DataAugmentForObjectDetection():
    def __init__(self, crop_rate=0.5, shift_rate=0.5, change_light_rate=0.5,  add_noise_rate=0.5,
                cutout_rate=0.5, cut_out_length=50, cut_out_holes=1, cut_out_threshold=0.5):
        self.crop_rate = crop_rate
        self.shift_rate = shift_rate
        self.change_light_rate = change_light_rate
        self.cutout_rate = cutout_rate
        self.add_noise_rate = add_noise_rate
        self.cut_out_length = cut_out_length
        self.cut_out_holes = cut_out_holes
        self.cut_out_threshold = cut_out_threshold

    # 高斯模糊
    def _addNoise(self, img):
        size = random.choice((5,9,11))
        return cv2.GaussianBlur(img, ksize=(size,size), sigmaX=0, sigmaY=0)

    
    # 调整亮度
    def _changeLight(self, img):
        flag = random.uniform(0.6, 1.3) #flag>1为调暗,小于1为调亮
        return exposure.adjust_gamma(img, flag)

    def _resever(self, img):
        ih, iw, ic = img.shape
        imagere = np.zeros((ih, iw, ic),dtype=np.uint8)
        for i in range(ic):
            ind = ic - i-1
            imagere[:,:,i] = img[:,:,ind]
        # imagere.astype(int)
        return imagere



    # cutout
    def _cutout(self, img, bboxes, length=100, n_holes=1, threshold=0.5):
        def cal_iou(boxA, boxB):
            xA = max(boxA[0], boxB[0])
            yA = max(boxA[1], boxB[1])
            xB = min(boxA[2], boxB[2])
            yB = min(boxA[3], boxB[3])

            if xB <= xA or yB <= yA:
                return 0.0
            interArea = (xB - xA + 1) * (yB - yA + 1)
            boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
            iou = interArea / float(boxBArea)
            return iou
        if img.ndim == 3:
            h,w,c = img.shape
        else:
            _,h,w,c = img.shape
        
        mask = np.ones((h,w,c), np.float32)

        for n in range(n_holes):
            chongdie = True  
            while chongdie:
                y = np.random.randint(h)
                x = np.random.randint(w)
                y1 = np.clip(y - length // 2, 0, h)
                y2 = np.clip(y + length // 2, 0, h)
                x1 = np.clip(x - length // 2, 0, w)
                x2 = np.clip(x + length // 2, 0, w)

                chongdie = False
                for box in bboxes:
                    if cal_iou([x1,y1,x2,y2], box) > threshold:
                        chongdie = True
                        break
            mask[y1: y2, x1: x2, :] = 0.
        img = img * mask
        return img

    # 裁剪
    def _crop_img_bboxes(self, img, bboxes):
        '''
        裁剪后的图片要包含所有的框
        输入:
            img:图像array
            bboxes:该图像包含的所有boundingboxs,一个list,每个元素为[x_min, y_min, x_max, y_max],要确保是数值
        输出:
            crop_img:裁剪后的图像array
            crop_bboxes:裁剪后的bounding box的坐标list
        '''
        #---------------------- 裁剪图像 ----------------------
        w = img.shape[1]
        h = img.shape[0]
        x_min = w   #裁剪后的包含所有目标框的最小的框
        x_max = 0
        y_min = h
        y_max = 0
        for bbox in bboxes:
            x_min = min(x_min, bbox[0])
            y_min = min(y_min, bbox[1])
            x_max = max(x_max, bbox[2])
            y_max = max(y_max, bbox[3])
        
        d_to_left = x_min           #包含所有目标框的最小框到左边的距离
        d_to_right = w - x_max      #包含所有目标框的最小框到右边的距离
        d_to_top = y_min            #包含所有目标框的最小框到顶端的距离
        d_to_bottom = h - y_max     #包含所有目标框的最小框到底部的距离

        #随机扩展这个最小框
        crop_x_min = int(x_min - random.uniform(0, d_to_left))
        crop_y_min = int(y_min - random.uniform(0, d_to_top))
        crop_x_max = int(x_max + random.uniform(0, d_to_right))
        crop_y_max = int(y_max + random.uniform(0, d_to_bottom))

        #确保不要越界
        crop_x_min = max(0, crop_x_min)
        crop_y_min = max(0, crop_y_min)
        crop_x_max = min(w, crop_x_max)
        crop_y_max = min(h, crop_y_max)

        crop_img = img[crop_y_min:crop_y_max, crop_x_min:crop_x_max]
        
        #---------------------- 裁剪boundingbox ----------------------
        #裁剪后的boundingbox坐标计算
        crop_bboxes = list()
        for bbox in bboxes:
            crop_bboxes.append([bbox[0]-crop_x_min, bbox[1]-crop_y_min, bbox[2]-crop_x_min, bbox[3]-crop_y_min])
        
        return crop_img, crop_bboxes
  
    # 平移
    def _shift_pic_bboxes(self, img, bboxes):
        #---------------------- 平移图像 ----------------------
        w = img.shape[1]
        h = img.shape[0]
        x_min = w   #裁剪后的包含所有目标框的最小的框
        x_max = 0
        y_min = h
        y_max = 0
        for bbox in bboxes:
            x_min = min(x_min, bbox[0])
            y_min = min(y_min, bbox[1])
            x_max = max(x_max, bbox[2])
            y_max = max(y_max, bbox[3])
        
        d_to_left = x_min           #包含所有目标框的最大左移动距离
        d_to_right = w - x_max      #包含所有目标框的最大右移动距离
        d_to_top = y_min            #包含所有目标框的最大上移动距离
        d_to_bottom = h - y_max     #包含所有目标框的最大下移动距离

        x = random.uniform(-(d_to_left-1) / 3, (d_to_right-1) / 3)
        y = random.uniform(-(d_to_top-1) / 3, (d_to_bottom-1) / 3)
        
        M = np.float32([[1, 0, x], [0, 1, y]])  #x为向左或右移动的像素值,正为向右负为向左; y为向上或者向下移动的像素值,正为向下负为向上
        shift_img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))

        #---------------------- 平移boundingbox ----------------------
        shift_bboxes = list()
        for bbox in bboxes:
            shift_bboxes.append([bbox[0]+x, bbox[1]+y, bbox[2]+x, bbox[3]+y])

        return shift_img, shift_bboxes

    # 镜像
    def _filp_pic_bboxes(self, img, bboxes):
        # ---------------------- 翻转图像 ----------------------
        import copy
        flip_img = copy.deepcopy(img)
        if random.random() < 0.5:    #0.5的概率水平翻转，0.5的概率垂直翻转
            horizon = True
        else:
            horizon = False
        h,w,_ = img.shape
        if horizon: #水平翻转
            flip_img =  cv2.flip(flip_img, 1)   #1是水平，-1是水平垂直
        else:
            flip_img = cv2.flip(flip_img, 0)

        # ---------------------- 调整boundingbox ----------------------
        flip_bboxes = list()
        for box in bboxes:
            x_min = box[0]
            y_min = box[1]
            x_max = box[2]
            y_max = box[3]
            if horizon:
                flip_bboxes.append([w-x_max, y_min, w-x_min, y_max])
            else:
                flip_bboxes.append([x_min, h-y_max, x_max, h-y_min])

        return flip_img, flip_bboxes


    # 旋转
    def _rotate_img_bbox(self, img, bboxes, angle=5, scale=1.):
        #---------------------- 旋转图像 ----------------------
        w = img.shape[1]
        h = img.shape[0]
        # 角度变弧度
        rangle = np.deg2rad(angle)  # angle in radians
        # now calculate new image width and height
        nw = (abs(np.sin(rangle)*h) + abs(np.cos(rangle)*w))*scale
        nh = (abs(np.cos(rangle)*h) + abs(np.sin(rangle)*w))*scale
        # ask OpenCV for the rotation matrix
        rot_mat = cv2.getRotationMatrix2D((nw*0.5, nh*0.5), angle, scale)
        # calculate the move from the old center to the new center combined
        # with the rotation
        rot_move = np.dot(rot_mat, np.array([(nw-w)*0.5, (nh-h)*0.5,0]))
        # the move only affects the translation, so update the translation
        # part of the transform
        rot_mat[0,2] += rot_move[0]
        rot_mat[1,2] += rot_move[1]
        # 仿射变换
        #rot_img = cv2.warpAffine(img, rot_mat, (int(math.ceil(nw)), int(math.ceil(nh))), flags=cv2.INTER_LANCZOS4)
        rot_img = np.zeros((int(math.ceil(nh)), int(math.ceil(nw)),25))
        for i  in range(25):
            rot_img_temp = img[:,:,i]
            rot_img_temp = cv2.warpAffine(rot_img_temp, rot_mat, (int(math.ceil(nw)), int(math.ceil(nh))), flags=cv2.INTER_LANCZOS4)
            rot_img[:,:,i] = rot_img_temp

        #---------------------- 矫正bbox坐标 ----------------------
        # rot_mat是最终的旋转矩阵
        # 获取原始bbox的四个中点，然后将这四个点转换到旋转后的坐标系下
        rot_bboxes = list()
        for bbox in bboxes:
            xmin = bbox[0]
            ymin = bbox[1]
            xmax = bbox[2]
            ymax = bbox[3]
            point1 = np.dot(rot_mat, np.array([(xmin+xmax)/2, ymin, 1]))
            point2 = np.dot(rot_mat, np.array([xmax, (ymin+ymax)/2, 1]))
            point3 = np.dot(rot_mat, np.array([(xmin+xmax)/2, ymax, 1]))
            point4 = np.dot(rot_mat, np.array([xmin, (ymin+ymax)/2, 1]))
            # 合并np.array
            concat = np.vstack((point1, point2, point3, point4))
            # 改变array类型
            concat = concat.astype(np.int32)
            # 得到旋转后的坐标
            rx, ry, rw, rh = cv2.boundingRect(concat)
            rx_min = rx
            ry_min = ry
            rx_max = rx+rw
            ry_max = ry+rh
            # 加入list中
            rot_bboxes.append([rx_min, ry_min, rx_max, ry_max])
        
        return rot_img, rot_bboxes


    def dataAugment(self, image, coords, num, file, xml_path,out_jpg_path):
        '''
        图像增强
        输入:
            img:图像array
            bboxes:该图像的所有框坐标
        输出:
            img:增强后的图像
            bboxes:增强后图片对应的box
        '''
        change_num = 0  #改变的次数
        print('------')

        option=[1,2,3,4,5,6,7,8,9,10,11,12,13]       #  选择数据增强的方式

    
        coords = parse_xml(xml_path)
        names = [coord[4] for coord in coords]
        coords = [coord[:4] for coord in coords]

        if 1 in option:
            num+=1
            print('裁剪')
            img, auged_bboxes = self._crop_img_bboxes(image, coords)
            [auged_bboxes[i].append(name) for i, name in enumerate(names)]
            ig = np.array(img[:,:,1].astype(np.uint8))
            cv2.imwrite(out_jpg_path+str(num)+".jpg",ig)        #    增强对应的3通道伪彩色
            img = np.transpose(img,(2,0,1)) # 相当于转置 (216,409,25)
            tr.imsave(os.path.join(out_path,str(num)+".tif"), img) 
            generate_xml(file, auged_bboxes, list(img.shape), out_xml_path,num)

        if 2 in option:
            num+=1
            print('平移')
            img, auged_bboxes1 = self._shift_pic_bboxes(image, coords)
            [auged_bboxes1[i].append(name) for i, name in enumerate(names)]
            ig = np.array(img[:,:,1].astype(np.uint8))
            cv2.imwrite(out_jpg_path+str(num)+".jpg",ig)
            img = np.transpose(img,(2,0,1)) 
            tr.imsave(os.path.join(out_path,str(num)+".tif"), img) 
            generate_xml(file, auged_bboxes1, list(img.shape), out_xml_path,num)

        if 3 in option:
            num+=1 #改变亮度
            print('亮度')
            img = self._changeLight(image)
            auged_bboxes3 = coords
            [auged_bboxes3[i].append(name) for i, name in enumerate(names)]
            ig = np.array(img[:,:,1].astype(np.uint8))
            cv2.imwrite(out_jpg_path+str(num)+".jpg",ig)
            img = np.transpose(img,(2,0,1)) 
            tr.imsave(os.path.join(out_path,str(num)+".tif"), img) 
            generate_xml(file, auged_bboxes3, list(img.shape), out_xml_path,num)

        
        if 4 in option:
            num+=1    #加噪声
            print('高斯模糊')
            img = self._addNoise(image)
            auged_bboxes4 = coords
            [auged_bboxes4[i].append(name) for i, name in enumerate(names)]
            ig = np.array(img[:,:,1].astype(np.uint8))
            cv2.imwrite(out_jpg_path+str(num)+".jpg",ig)
            img = np.transpose(img,(2,0,1)) 
            tr.imsave(os.path.join(out_path,str(num)+".tif"), img) 
            generate_xml(file, auged_bboxes4, list(img.shape), out_xml_path,num)



        if 5 in option:
            num+=1  #cutout
            print('剪切')
            change_num += 1
            img = self._cutout(image, coords, length=self.cut_out_length, n_holes=self.cut_out_holes,
                                threshold=self.cut_out_threshold)
            auged_bboxes5 = coords
            [auged_bboxes5[i].append(name) for i, name in enumerate(names)]
            ig = np.array(img[:,:,1].astype(np.uint8))
            cv2.imwrite(out_jpg_path+str(num)+".jpg",ig)
            img = np.ascontiguousarray(np.transpose(img,(2,0,1))) 
            tr.imsave(os.path.join(out_path,str(num)+".tif"), img.astype(np.uint8)) 
            generate_xml(file, auged_bboxes5, list(img.shape), out_xml_path,num)
        if 6 in option:    #翻转
            print('翻转')
            num+=1  
            img, auged_bboxes6 = self._filp_pic_bboxes(image, coords)
            [auged_bboxes6[i].append(name) for i, name in enumerate(names)]
            ig = np.array(img[:,:,1].astype(np.uint8))
            cv2.imwrite(out_jpg_path+str(num)+".jpg",ig)
            img = np.transpose(img,(2,0,1)) 
            tr.imsave(os.path.join(out_path,str(num)+".tif"), img) 
            generate_xml(file, auged_bboxes6, list(img.shape), out_xml_path,num)

        if 7 in option:   # 通道倒置
            num += 1  # cutout
            print('通道倒置')
            img = self._resever(image)
            auged_bboxes7 = coords
            [auged_bboxes7[i].append(name) for i, name in enumerate(names)]
            ig = np.array(img[:,:,1].astype(np.uint8))
            cv2.imwrite(out_jpg_path+str(num)+".jpg",ig)
            img = np.transpose(img,(2,0,1))
            tr.imsave(os.path.join(out_path,str(num)+".tif"), img)
            generate_xml(file, auged_bboxes7, list(img.shape), out_xml_path,num)

        if 8 in option:     #旋转
            print('旋转5')
            num += 1
            # angle = random.uniform(-self.max_rotation_angle, self.max_rotation_angle)
            angle = 5
            scale = random.uniform(0.7, 0.8)
            img, auged_bboxes7 = self._rotate_img_bbox(image, coords, angle, scale)
            [auged_bboxes7[i].append(name) for i, name in enumerate(names)]
            ig = np.array(img[:,:,1].astype(np.uint8))
            cv2.imwrite(out_jpg_path+str(num)+".jpg",ig)
            img = np.transpose(img,(2,0,1))
            tr.imsave(os.path.join(out_path,str(num)+".tif"), img.astype(np.uint8))
            generate_xml(file, auged_bboxes7, list(img.shape), out_xml_path,num)

        if 9 in option:     #旋转
            print('旋转10')
            num += 1
            # angle = random.uniform(-self.max_rotation_angle, self.max_rotation_angle)
            angle = 10
            scale = random.uniform(0.7, 0.8)
            img, auged_bboxes7 = self._rotate_img_bbox(image, coords, angle, scale)
            [auged_bboxes7[i].append(name) for i, name in enumerate(names)]
            ig = np.array(img[:,:,1].astype(np.uint8))
            cv2.imwrite(out_jpg_path+str(num)+".jpg",ig)
            img = np.transpose(img,(2,0,1))
            tr.imsave(os.path.join(out_path,str(num)+".tif"), img.astype(np.uint8))
            generate_xml(file, auged_bboxes7, list(img.shape), out_xml_path,num)

        if 10 in option:     #旋转
            print('旋转15')
            num += 1
            # angle = random.uniform(-self.max_rotation_angle, self.max_rotation_angle)
            angle = 15
            scale = random.uniform(0.7, 0.8)
            img, auged_bboxes7 = self._rotate_img_bbox(image, coords, angle, scale)
            [auged_bboxes7[i].append(name) for i, name in enumerate(names)]
            ig = np.array(img[:,:,1].astype(np.uint8))
            cv2.imwrite(out_jpg_path+str(num)+".jpg",ig)
            img = np.transpose(img,(2,0,1))
            tr.imsave(os.path.join(out_path,str(num)+".tif"), img.astype(np.uint8))
            generate_xml(file, auged_bboxes7, list(img.shape), out_xml_path,num)

        if 11 in option:     #旋转
            print('旋转-5')
            num += 1
            # angle = random.uniform(-self.max_rotation_angle, self.max_rotation_angle)
            angle = 355
            scale = random.uniform(0.7, 0.8)
            img, auged_bboxes7 = self._rotate_img_bbox(image, coords, angle, scale)
            [auged_bboxes7[i].append(name) for i, name in enumerate(names)]
            ig = np.array(img[:,:,1].astype(np.uint8))
            cv2.imwrite(out_jpg_path+str(num)+".jpg",ig)
            img = np.transpose(img,(2,0,1))
            tr.imsave(os.path.join(out_path,str(num)+".tif"), img.astype(np.uint8))
            generate_xml(file, auged_bboxes7, list(img.shape), out_xml_path,num)

        if 12 in option:     #旋转
            print('旋转-10')
            num += 1
            # angle = random.uniform(-self.max_rotation_angle, self.max_rotation_angle)
            angle = 350
            scale = random.uniform(0.7, 0.8)
            img, auged_bboxes7 = self._rotate_img_bbox(image, coords, angle, scale)
            [auged_bboxes7[i].append(name) for i, name in enumerate(names)]
            ig = np.array(img[:,:,1].astype(np.uint8))
            cv2.imwrite(out_jpg_path+str(num)+".jpg",ig)
            img = np.transpose(img,(2,0,1))
            tr.imsave(os.path.join(out_path,str(num)+".tif"), img.astype(np.uint8))
            generate_xml(file, auged_bboxes7, list(img.shape), out_xml_path,num)

        if 13 in option:     #旋转
            print('旋转-15')
            num += 1
            # angle = random.uniform(-self.max_rotation_angle, self.max_rotation_angle)
            angle = 345
            scale = random.uniform(0.7, 0.8)
            img, auged_bboxes7 = self._rotate_img_bbox(image, coords, angle, scale)
            [auged_bboxes7[i].append(name) for i, name in enumerate(names)]
            ig = np.array(img[:,:,1].astype(np.uint8))
            cv2.imwrite(out_jpg_path+str(num)+".jpg",ig)
            img = np.transpose(img,(2,0,1))
            tr.imsave(os.path.join(out_path,str(num)+".tif"), img.astype(np.uint8))
            generate_xml(file, auged_bboxes7, list(img.shape), out_xml_path,num)

        # if 7 in option:     #旋转
        #     print('旋转')
        #     num += 1
        #     # angle = random.uniform(-self.max_rotation_angle, self.max_rotation_angle)
        #     angle = random.sample([90, 180, 270],1)[0]
        #     scale = random.uniform(0.7, 0.8)
        #     img, auged_bboxes7 = self._rotate_img_bbox(image, coords, angle, scale)
        #     [auged_bboxes7[i].append(name) for i, name in enumerate(names)]
        #     ig = np.array(img[:,:,1].astype(np.uint8))
        #     cv2.imwrite(out_jpg_path+str(num)+".jpg",ig)
        #     img = np.transpose(img,(2,0,1))
        #     tr.imsave(os.path.join(out_path,str(num)+".tif"), img.astype(np.uint8))
        #     generate_xml(file, auged_bboxes7, list(img.shape), out_xml_path,num)
        print('\n')
        return num

def normalgray(image):
    ih, iw, ic = image.shape
    msnormal = np.zeros((ih, iw, ic)).astype(np.uint8)
    for i in range(ic):
        midimage = image[:,:,i]
        msmin = midimage.min()
        msmax = midimage.max()
        msnormal[:,:,i] = (((midimage-msmin) / (msmax-msmin))*255).astype(np.uint8)
    return msnormal

if __name__ == '__main__':
    need_aug_num = 1
    dataAug = DataAugmentForObjectDetection()
    # source_pic_root_path = r'./VOC2007/potatoes'   #  待增强.tif
    source_pic_root_path = r'D:\研究成果\Paper 8 Improved YOLOv3-tiny\code and data\tiny-yolov3-potatoes-fpg\VOC2007\Potatoes\Potatoes'
    source_xml_root_path = r'D:\研究成果\Paper 8 Improved YOLOv3-tiny\code and data\tiny-yolov3-potatoes-fpg\VOC2007\Potatoes\Annotations'     #  待增强.tif 对应.xml
    out_path=r"D:\研究成果\Paper 8 Improved YOLOv3-tiny\code and data\tiny-yolov3-potatoes-fpg\VOC2007\Augdata\Augtatoes"              #  生成的增强.tif
    out_jpg_path=r"C:\Users\c508\Desktop\potatoesjpg\\"              #  生成的增强伪彩色 .jpg
    out_xml_path=r"D:\研究成果\Paper 8 Improved YOLOv3-tiny\code and data\tiny-yolov3-potatoes-fpg\VOC2007\Augdata\AugAnnotations"            #  生成的增强.tif 对应.xml
    '''
    #img1 = cv2.imread('./data4aug/image/c01_New-1.jpg')  # (216, 409, 3) 230 6
    img = tr.imread('./data4aug/augimage/11.tif')   # (25, 216, 409) 433.31766 -35.991863
    img = np.transpose(img,(1,2,0)) # 相当于转置 (216,409,25)
    #image = normalgray(image)
    print(np.shape(img), np.max(img), np.min(img))
    for i in range (25):
        cv2.imshow("de",img[:,:,i])
        cv2.waitKey(100)
    '''
    #print(np.shape(image), np.max(image[:,:,1]), np.min(image[:,:,1]))
    all_files = os.listdir(source_pic_root_path)
    num = len(all_files)  # 原数据量
    print("num:  ",num)
    
    start=time.perf_counter()
    for file in os.listdir(source_pic_root_path):
        if file.endswith(".tif") or file.endswith(".TIF"):
            pic_path = os.path.join(source_pic_root_path, file)
            xml_path = os.path.join(source_xml_root_path, file[:-4]+'.xml')
            coords = parse_xml(xml_path)
            img = tr.imread(pic_path)   # (25, 216, 409) 433.31766 -35.99186
            image = np.ascontiguousarray(np.transpose(img,(1,2,0))) # 相当于转置 (216,409,25) 0-255
            image = normalgray(image)
            names = [coord[4] for coord in coords]
            coords = [coord[:4] for coord in coords]
            num = dataAug.dataAugment(image, coords, num, file, xml_path,out_jpg_path)
    
    end = time.perf_counter()
    print("time:{}".format(end-start))


