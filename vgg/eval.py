"""
Simple tester for the vgg19_trainable
"""

import os, sys
from random import shuffle
import tensorflow as tf

import vgg19_trainable as vgg19
import utils
import numpy as np
import cv2
import json

BATCHSIZE = 32
DATAPATH = '/beegfs/ua349/flower_photos'
IMSIZE = 224
CDIM = 3
NETSIZE = 1000

def get_image(impath):
	img = utils.load_image(impath)
	img = img[:IMSIZE, :IMSIZE, :] # just crop to keep ratio if possible
	xscale = float(IMSIZE) / img.shape[1]
	yscale = float(IMSIZE) / img.shape[0]
	img = cv2.resize(img, (0, 0), fx=xscale, fy=yscale)
	return img

folders = [fl for fl in os.listdir(DATAPATH) if '.' not in fl]
metadata = {}
lookup = {}
for fii, fl in enumerate(folders):
        if fl not in lookup: lookup[fl] = fii
        metadata[fl] = ['%s/%s/%s' % (DATAPATH, fl, img) for img in os.listdir('%s/%s' % (DATAPATH, fl)) if '.jpg' in img]
        #for ii in range(len(metadata[fl])):
        #       datainds.append((fl, ii))
        #print(fl, len(metadata[fl]))

with open('evaldata.json') as fl:
    dtest = json.load(fl)


sess = tf.Session()
images = tf.placeholder(tf.float32, [BATCHSIZE, IMSIZE, IMSIZE, CDIM])
true_out = tf.placeholder(tf.float32, [BATCHSIZE, NETSIZE])
train_mode = tf.placeholder(tf.bool)

vgg = vgg19.Vgg19('./checkpoint.npy')
vgg.build(images, train_mode)

# print number of variables used: 143667240 variables, i.e. ideal size = 548MB
print('Trainable vars:', vgg.get_var_count())

sess.run(tf.global_variables_initializer())
cost = tf.reduce_sum((vgg.prob - true_out) ** 2)
train = tf.train.GradientDescentOptimizer(0.0001).minimize(cost)
numbatches = int(len(dtest) / BATCHSIZE)
correct = 0
tally = 0
for bii in range(numbatches):
	batchinds = dtest[bii*BATCHSIZE:(bii+1) * BATCHSIZE]
	if len(batchinds) < BATCHSIZE: continue
	batch = np.array([get_image(metadata[fl][ind]) for fl, ind in batchinds])
	labels = [[1.0 if lookup[fl] == ii else 0.0 for ii in range(NETSIZE)] for fl, _ in batchinds]
	prob = sess.run(vgg.prob, feed_dict={images: batch, train_mode: False})
	for ii, ent in enumerate(prob):
		if np.argmax(ent) == lookup[batchinds[ii][0]]:
			correct += 1
		tally += 1

print('%d/%d = %.2f' % (correct, tally, correct / tally))
