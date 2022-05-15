# MDDNet
Train MDDNet model

(1) Run "data_augmentation" to expand the dataset.

(2) Class and preset anchor size should be rewritten "voc-classes" and "tiny_yolo_anchors" in "model_data".

(3) Adjust our data to the VOC2007 format. 

(4) Run "VOC2007\test" and "voc_annotation" sequentially to obtain train, val, and test dataset.

(5) Train MDDNet model using "tiny_train".

(6) Test model using "yolo_test_vatch".

Note that: Adjust the Lines 90-93 of "tiny_train" can train YOLOv3, YOLOv3-tiny, and DY3TNet model.

# Requirement
Keras == 2.2.4

Tensorflow == 1.13.1

# Citation
If you find this work or code is helpful in your research, please cite:

Yu Yang, Zhenfang Liu, Min Huang, Qibing Zhu, Xin Zhao. Detection and identification of multi-type defects on potatoes using multispectral imaging combined with a deep learning model.

# Acknowledgments
Tiny-yolov3 (Keras): https://github.com/Eatzhy/tiny-yolov3

Res2Net (Keras): https://github.com/fupiao1998/res2net-keras/blob/master/res2net.py

