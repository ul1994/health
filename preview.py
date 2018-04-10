from __future__ import print_function
# from ljpeg import ljpeg
import numpy as np
import json, sys, os

import time
from shutil import copyfile

import matplotlib.pyplot as plt
import scipy.misc
import cv2
import scipy.ndimage.filters as fi
import numpy as np
import h5py

if __name__ == '__main__':
	plt.ion()

	labels = [
		'cancers',
		'benigns',
		'callbacks',
		'normals',
	]

	DATAROOT = '/media/ul1994/ssd1tb/phenotype-tiles'
	DIAG = 'cancers'
	# INSPECT = 'A_1043_1.LEFT_CC'
	# INSPECT = 'B_3511_1.LEFT_CC'
	# INSPECT = 'B_3504_1.RIGHT_CC'
	INSPECT = 'A_1127_1.RIGHT_CC'

	fpath ='%s/%s/%s' % (DATAROOT, DIAG, INSPECT)
	odpath = fpath + '.npy'
	maskpath = fpath + '-mask.npy'
	mask = np.load(maskpath)
	density = np.load(odpath)
	density = density / 3.6
	density[density < 0] = 0
	density[density > 1] = 1

	plt.figure(figsize=(14, 10))
	plt.subplot(1, 3, 1)
	plt.imshow(density)
	plt.subplot(1, 3, 2)
	plt.imshow(mask)
	plt.subplot(1, 3, 3)
	smoothed = fi.gaussian_filter(mask.astype(np.float32), 2)
	# print(np.max(smoothed))
	plt.imshow(smoothed)
	plt.tight_layout()
	plt.show()

	random_db = h5py.File('%s-random.h5' % fpath, 'r')
	cancer_db = h5py.File('%s-samples.h5' % fpath, 'r')

	plt.figure(figsize=(14, 10))
	for ii, density in enumerate(random_db.get('density')):
		plt.subplot(4, 8, 2*ii + 1)
		plt.imshow(density)
		mask = random_db.get('mask')[ii]
		plt.subplot(4, 8, 2*ii + 2)
		plt.imshow(mask)
	plt.tight_layout()
	plt.show()

	plt.figure(figsize=(14, 10))
	for ii, density in enumerate(cancer_db.get('density')):
		plt.subplot(4, 8, 2*ii + 1)
		plt.imshow(density)
		mask = cancer_db.get('mask')[ii]
		plt.subplot(4, 8, 2*ii + 2)
		plt.imshow(mask)
	plt.tight_layout()
	plt.show()

	# plt.figure(figsize=(14, 10))
	# for ii, fname in enumerate([fl for fl in cancer_samples if 'mask' not in fl]):
	# 	plt.subplot(4, 4, ii + 1)
	# 	bpath = '%s/%s/%s' % (DATAROOT, DIAG, fname)
	# 	blk = np.load(bpath)
	# 	print ('2', blk.dtype)
	# 	plt.imshow(blk)

	# plt.tight_layout()
	# plt.show()

	# plt.figure(figsize=(14, 10))
	# for yy in range(4):
	# 	for xx in range(4):
	# 		plt.subplot(4, 4, yy * 4 + xx + 1)
	# 		bpath = '%s/%s/%s-%dx%d' % (DATAROOT, DIAG, INSPECT, yy, xx)
	# 		blk = np.load(bpath + '-mask.npy')
	# 		blk = fi.gaussian_filter(blk.astype(np.float32), 13)
	# 		plt.imshow(blk)

	# plt.tight_layout()
	# plt.show()

	# plt.figure(figsize=(14, 10))
	# for ii in range(16):
	# 	plt.subplot(4, 4, ii +  1)
	# 	bpath = '%s/%s/%s-sample-%d' % (DATAROOT, DIAG, INSPECT, ii)
	# 	blk = np.load(bpath + '-mask.npy')
	# 	print(blk.shape)
	# 	blk = fi.gaussian_filter(blk.astype(np.float32), 13)
	# 	plt.imshow(blk)

	# plt.figure(figsize=(14, 10))
	# for ii in range(16):
	# 	plt.subplot(4, 4, ii +  1)
	# 	bpath = '%s/%s/%s-sample-%d' % (DATAROOT, DIAG, INSPECT, ii)
	# 	blk = np.load(bpath + '.npy')
	# 	plt.imshow(blk)

	# plt.tight_layout()
	plt.show()

	try: input()
	except: pass
