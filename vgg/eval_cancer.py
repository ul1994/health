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
DATAPATH = '/beegfs/ua349/data/tissue128/joined'
IMSIZE = 256
CDIM = 3
NETSIZE = 1000

def get_image(impath):
	img = utils.load_image(impath, imsize=IMSIZE)
	img = img[:IMSIZE, :IMSIZE, :] # just crop to keep ratio if possible
	return img

datainds = []
metadata = {}
lookup = { 'normals': 0, 'benigns': 1, 'cancers': 2 }
imgfiles = [fl for fl in os.listdir(DATAPATH) if '.png' in fl]
for fii, fl in enumerate(imgfiles):
	rawlabel = fl.split('-')[-1].replace('.png', '')
	if rawlabel not in metadata: metadata[rawlabel] = []
	metadata[rawlabel].append('%s/%s' % (DATAPATH, fl))

with open('evaldata.json') as fl:
    dtest = json.load(fl)

sess = tf.Session()
images = tf.placeholder(tf.float32, [BATCHSIZE, IMSIZE, IMSIZE, CDIM])
true_out = tf.placeholder(tf.float32, [BATCHSIZE, NETSIZE])
train_mode = tf.placeholder(tf.bool)

vgg = vgg19.Vgg19('./checkpoint-cancer.npy')
vgg.build(images, train_mode, imsize=IMSIZE)

sess.run(tf.global_variables_initializer())
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
