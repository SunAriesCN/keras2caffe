#!/usr/bin/env python
import caffe
import cv2
import numpy as np
from keras.preprocessing import image
#testing the model
#caffe.set_mode_gpu()
net  = caffe.Net('squ.prototxt', 'squ.caffemodel', caffe.TEST)

#img = cv2.imread('images/cat.jpeg')
#img = cv2.resize(img, (227, 227))
#img = img[...,::-1]  #RGB 2 BGR

img = image.load_img('./images/cat.jpeg', target_size=(227, 227))
data = image.img_to_array(img)
data = data.transpose((2, 0, 1))
data = np.expand_dims(data, axis=0)

#data = np.array(img, dtype=np.float32)
#data = data.transpose((2, 0, 1))
##data.shape = (1,) + data.shape

net.blobs['data'].data[...] = data

out = net.forward()
print out.keys()[0]
print out['pool3'].shape

pred = np.array(out['pool3'].transpose((0, 2, 3, 1 )))
print pred.shape
pred = pred.reshape(-1)
print pred

#print out['pool3'].transpose((0, 3, 2, 1 )) == out['pool3'].transpose((0, 2, 3, 1 ))
