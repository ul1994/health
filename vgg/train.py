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

TASK = sys.argv[1]

def get_image(impath):
	img = utils.load_image(impath, imsize=IMSIZE)
	#img = img[:IMSIZE, :IMSIZE, :] # just crop to keep ratio if possible
	# xscale = float(IMSIZE) / img.shape[1]
	# yscale = float(IMSIZE) / img.shape[0]
	# img = cv2.resize(img, (0, 0), fx=xscale, fy=yscale)
	return img

if TASK == 'cancer':
	EPOCHS = 64
	# DATAPATH = '/beegfs/ua349/data/tissue128/MLO'
	DATAPATH = '/beegfs/ua349/data/tissue256/CC'
	BATCHSIZE = 32
	DATASPLIT = 0.9
	IMSIZE = 224
	CDIM = 3

	datainds = []
	folders = [fl for fl in os.listdir(DATAPATH) if '.' not in fl]
	metadata = {}
	lookup = {}
	for fii, fl in enumerate(folders):
		if fl not in lookup: lookup[fl] = fii
		metadata[fl] = ['%s/%s/%s' % (DATAPATH, fl, img) for img in os.listdir('%s/%s' % (DATAPATH, fl)) if '.png' in img]
		metadata[fl] = metadata[fl][:1000]
		for ii in range(len(metadata[fl])):
			datainds.append((fl, ii))
		print(fl, len(metadata[fl]))
	shuffle(datainds)
	OUTSIZE = len(folders)

elif TASK == 'flower':
	EPOCHS = 32
	DATAPATH = '/beegfs/ua349/flower_photos'
	BATCHSIZE = 32
	DATASPLIT = 0.9
	IMSIZE = 224
	CDIM = 3

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
	OUTSIZE = len(folders)

splitat = int(len(datainds)*DATASPLIT)
dtrain, dtest = datainds[:splitat], datainds[splitat:]
print('Train/Test Split:', len(dtrain), len(dtest))

import json
with open('evaldata-cancer.json', 'w') as fl:
        json.dump(dtest, fl)

sess = tf.Session()
images = tf.placeholder(tf.float32, [BATCHSIZE, IMSIZE, IMSIZE, CDIM])
true_out = tf.placeholder(tf.float32, [BATCHSIZE, OUTSIZE])
train_mode = tf.placeholder(tf.bool)

vgg = vgg19.Vgg19('./vgg19.npy')
vgg.build(images, train_mode, imsize=IMSIZE, outsize=OUTSIZE)

# print number of variables used: 143667240 variables, i.e. ideal size = 548MB
print('Trainable vars:', vgg.get_var_count())

def evaluate():
	numbatches = int(len(dtest) / BATCHSIZE)
	correct = 0
	tally = 0
	maxpred = np.zeros(OUTSIZE)
	for bii in range(numbatches):
		batchinds = dtest[bii*BATCHSIZE:(bii+1) * BATCHSIZE]
		if len(batchinds) < BATCHSIZE: continue
		batch = np.array([get_image(metadata[fl][ind]) for fl, ind in batchinds])
		# labels = [[1.0 if lookup[fl] == ii else 0.0 for ii in range(OUTSIZE)] for fl, _ in batchinds]
		prob = sess.run(vgg.prob, feed_dict={images: batch, train_mode: False})
		for ii, ent in enumerate(prob):
			maxpred[np.argmax(ent)] += 1
			if np.argmax(ent) == lookup[batchinds[ii][0]]:
				correct += 1
			tally += 1

	maxpred /= np.sum(maxpred)
	distrib = ' '.join(['%.1f' % val for val in maxpred])
	print('%d/%d = %.2f   (%s)' % (correct, tally, correct / tally, distrib))

sess.run(tf.global_variables_initializer())
cost = tf.reduce_sum((vgg.prob - true_out) ** 2)
train = tf.train.GradientDescentOptimizer(0.0001).minimize(cost)
numbatches = int(len(dtrain) / BATCHSIZE)

evaluate()

for epochi in range(EPOCHS):
	print('Epoch: %d/%d' %(epochi, EPOCHS))
	for bii in range(numbatches):
		sys.stdout.write('%d/%d\r' % (bii, numbatches))
		sys.stdout.flush()
		batchinds = dtrain[bii*BATCHSIZE:(bii+1) * BATCHSIZE]
		if len(batchinds) < BATCHSIZE: continue
		batch = np.array([get_image(metadata[fl][ind]) for fl, ind in batchinds])
		labels = [[1.0 if lookup[fl] == ii else 0.0 for ii in range(OUTSIZE)] for fl, _ in batchinds]
		# print(labels[0][:10])
		# print(labels[1][:10])
		# input()
		sess.run(train, feed_dict={images: batch, true_out: labels, train_mode: True})
	print()

	evaluate()
	#evalbatch = np.array([get_image(metadata[fl][ind]) for fl, ind in dtest])
	#evalres = sess.run(vgg.prob, feed_dict={images: evalbatch, train_mode: False})
	#for

# test classification again, should have a higher probability about tiger
# prob = sess.run(vgg.prob, feed_dict={images: batch1, train_mode: False})
# utils.print_prob(prob[0], './synset.txt')

vgg.save_npy(sess, './checkpoint-%s.npy' % TASK)
