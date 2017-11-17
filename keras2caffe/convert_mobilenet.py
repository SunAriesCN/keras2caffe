import caffe
import cv2
import numpy as np

from keras.applications.mobilenet import MobileNet

import keras2caffe


#converting

keras_model = MobileNet(weights='imagenet', include_top=True)

keras2caffe.convert(keras_model, 'mobilenet.prototxt', 'mobilenet.caffemodel')


#testing the model

caffe.set_mode_gpu()
net  = caffe.Net('mobilenet.prototxt', 'mobilenet.caffemodel', caffe.TEST)

img = cv2.imread('bear.jpg')
img = cv2.resize(img, (224, 224))
img = img[...,::-1]  #RGB 2 BGR

data = np.array(img, dtype=np.float32)
data = data.transpose((2, 0, 1))
data.shape = (1,) + data.shape

data -= 128
data /= 128

net.blobs['data'].data[...] = data

out = net.forward()
pred = out['predictions']

#softmax function
pred = np.exp(pred - np.max(pred))
pred /= pred.sum()

prob = np.max(pred)
cls = pred.argmax()

lines=open('synset_words.txt').readlines()
print prob, cls, lines[cls]

