import caffe
import cv2
import numpy as np

from keras.applications.inception_v3 import InceptionV3

import keras2caffe


img_path = 'bear.jpg'
keras_model = InceptionV3(weights='imagenet', include_top=True)
net  = caffe.Net('deploy.prototxt', 'InceptionV3.caffemodel', caffe.TEST)

###caffe
img = cv2.imread(img_path)
img = cv2.resize(img, (299, 299))
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
print 'caffe:\n', prob, cls, lines[cls]

###keras
base_img = cv2.imread(img_path)
base_img = cv2.resize(base_img, (299, 299))
base_img = base_img[...,::-1]

data = np.array(base_img, dtype=np.float32)
#data = data.transpose((2, 0, 1))
data.shape = (1,) + data.shape

data -= 128
data /= 128

pred = keras_model.predict(data)
#softmax function
#pred = np.exp(pred - np.max(pred))
#pred /= pred.sum()

prob = np.max(pred)
cls = pred.argmax()

lines=open('synset_words.txt').readlines()
print 'keras:\n', prob, cls, lines[cls]


