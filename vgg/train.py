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

DATAPATH = '../data/flower_photos'
BATCHSIZE = 32
DATASPLIT = 0.9
IMSIZE = 224
CDIM = 3
NETSIZE = 1000

def get_image(impath):
	img = utils.load_image(impath)
	img = img[:IMSIZE, :IMSIZE, :] # just crop to keep ratio if possible
	xscale = img.shape[1] / float(IMSIZE)
	yscale = img.shape[0] / float(IMSIZE)
	img = cv2.resize(img, (0, 0), fx=xscale, fy=yscale)
	return img

datainds = []
folders = [fl for fl in os.listdir(DATAPATH) if '.' not in fl]
metadata = {}
lookup = {}
for fii, fl in enumerate(folders):
	if fl not in lookup: lookup[fl] = fii
	metadata[fl] = ['%s/%s/%s' % (DATAPATH, fl, img) for img in os.listdir('%s/%s' % (DATAPATH, fl)) if '.jpg' in img]
	for ii in range(len(metadata[fl])):
		datainds.append((fl, ii))
	print(fl, len(metadata[fl]))
shuffle(datainds)

splitat = int(len(datainds)*DATASPLIT)
dtrain, dtest = datainds[:splitat], datainds[splitat:]

sess = tf.Session()
images = tf.placeholder(tf.float32, [BATCHSIZE, IMSIZE, IMSIZE, CDIM])
true_out = tf.placeholder(tf.float32, [BATCHSIZE, NETSIZE])
train_mode = tf.placeholder(tf.bool)

vgg = vgg19.Vgg19('./vgg19.npy')
vgg.build(images, train_mode)

# print number of variables used: 143667240 variables, i.e. ideal size = 548MB
print('Trainable vars:', vgg.get_var_count())

sess.run(tf.global_variables_initializer())
cost = tf.reduce_sum((vgg.prob - true_out) ** 2)
train = tf.train.GradientDescentOptimizer(0.0001).minimize(cost)
numbatches = int(len(dtrain) / BATCHSIZE)
for epochi in range(1):
	print('Epoch:', epochi)
	for bii in range(numbatches):
		sys.stdout.write('%d/%d\r' % (bii, numbatches))
		sys.stdout.flush()
		batchinds = dtrain[bii*BATCHSIZE:(bii+1) * BATCHSIZE]
		if len(batchinds) < BATCHSIZE: continue
		batch = np.array([get_image(metadata[fl][ind]) for fl, ind in batchinds])
		# print(type(batch))
		# input()
		labels = [[1.0 if lookup[fl] == ii else 0.0 for ii in range(NETSIZE)] for fl, _ in batchinds]
		sess.run(train, feed_dict={images: batch, true_out: labels, train_mode: True})
	print()

# test classification again, should have a higher probability about tiger
# prob = sess.run(vgg.prob, feed_dict={images: batch1, train_mode: False})
# utils.print_prob(prob[0], './synset.txt')

# test save
vgg.save_npy(sess, './test-save.npy')
