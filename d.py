

import math
import numpy as np
anchor_mask = [[6,7,8], [3,4,5], [0,1,2]]
a = [[1,2,3,4,5],[6,7,8,9,0]]
a = np.array(a)
b = a[...,0:2]
c = a[...,2:4]
num_anchors =6
num_classes = 8
h = 216
w = 409
shape=(h//{0:32, 1:16}[0], w//{0:32, 1:16}[0], num_anchors//3, num_classes+5)
print("输入大小：：：      ",math.ceil(h/{0:32, 1:16}[0]),h//{0:32, 1:16}[0],math.ceil(h/{0:32, 1:16}[1]),h//{0:32, 1:16}[1])         #       (?, ?, ?, 25) (2,)
#print("输入大小：：：      ",h,w,shape,h//{0:32, 1:16}[0],h//{0:32, 1:16}[1])         #       (?, ?, ?, 25) (2,)

from PIL import Image
img = Image.new('RGB', (125, 125))
img.show()

img1 = Image.new('RGB', (10,10), (128,128)) 
img1.show()