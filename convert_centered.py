from __future__ import print_function
from ljpeg import ljpeg
import numpy as np
import json, sys, os

import time
from shutil import copyfile
import cv2

from scipy.stats import multivariate_normal
import scipy.ndimage.filters as fi
import cPickle as pkl
import h5py
from blacklisted import BLACKLISTED
from utils import *

# DATAROOT = 'data'
DATAROOT = '/media/ul1994/ssd1tb/phenotype'

import matplotlib.pyplot as plt
import scipy.misc
import cv2
import random
import PIL

DEBUGONE = None
DEBUGANGLE = None
# DEBUGONE = 'A-1041-1'
DEBUGANGLE = 'RIGHT_CC'
DEBUGONE = 'B-3135-1'
# DEBUGONE = 'A-1127-1'
# DEBUGONE = 'A-1016-1'
# DEBUGANGLE = 'LEFT_CC'

if __name__ == '__main__':
	labels = [
		'cancers',
		'benigns',
		'callbacks',
		'normals',
	]

	with open('metadata.json') as fl:
		metadata = json.load(fl)

	if len(sys.argv) == 1: raise Exception('Specify label')


	lid = int(sys.argv[1])
	angle = 'CC'
	# angle = sys.argv[2]
	skip = 0
	label = labels[lid]

	chosen = [elem for elem in metadata.values() if elem['diagnosis'] == label]

	print('Parsing:', label, len(chosen))

	for ii, elem in enumerate(chosen):
		elem['root'] = elem['root'].replace('ssd1tb1', 'ssd1tb')
		t0 = time.time()
		try:
			assert len(elem['scans']) == 4
		except:
			print(elem['root'])
			print(elem['scans'])
			raise Exception('Few scans... %d' % (len(elem['scans'])))
		if DEBUGONE and elem['name'] != DEBUGONE:
			continue
		if ii < skip: continue
		# print()
		# print(elem['name'])

		available_scans = elem['scans'].values()
		if DEBUGANGLE:
			available_scans =[elem['scans'][DEBUGANGLE]]
		for ssii, scaninfo in enumerate(available_scans):
			if elem['name'] in BLACKLISTED:
				if scaninfo['name'].split('.')[1] in BLACKLISTED[elem['name']]:
					print()
					print('Skipping blacklisted:', elem['name'], scaninfo['name'])
					continue
			if angle not in scaninfo['name']: continue
			scanfile = '%s/%s.LJPEG' % (elem['root'], scaninfo['name'])
			try:
				scan = ljpeg.read(scanfile).astype(np.float)
			except:
				print('READ ERROR:', scaninfo['name'], elem['root'])
				continue
			scan = scan.reshape((max(scan.shape), min(scan.shape)))

			hh, ww = scan.shape
			calibrated = None
			exec('calibrated = %s(scan)' % elem['digitizer'].replace('-', '_'))
			maxod = 3.6
			calibrated = maxod - calibrated
			rawod = calibrated.copy()

			cliplow = maxod / 4.0
			calibrated -= cliplow
			calibrated /= (maxod - cliplow)
			calibrated[calibrated < 0] = 0
			calibrated[calibrated > 1] = 1
			png = (255 * calibrated).astype(np.uint8)

			# cleanup
			__png = trim(png)
			facing = direction(__png)
			trimmed_od = trim(rawod)
			if facing == 'right':
				__trimmed = cv2.flip(__png, 1)
				trimmed_od = cv2.flip(trimmed_od, 1)
			filled_od = canvas(trimmed_od)

			__filled = canvas(__trimmed)
			x0, y0, xf, yf = bbox(__filled)

			outmask_preview = None
			outmask = None
			outmarked = None
			cancer_tiles = []
			if 'overlays' in scaninfo:
				for overlay in scaninfo['overlays']:
					if not ('MALIGNANT' in overlay['pathology'] or 'BENIGN' in overlay['pathology']):
						# skip all non provens
						continue

					markedim = calibrated.copy()
					all_masks = []
					for jj, markup in enumerate(overlay['outlines']):
						mask, coords, minmax = applyMark(scaninfo['name'], markedim.shape, markup)
						all_masks.append((mask, coords, minmax))

					mask_i = 0
					keep_masks = np.array([kk for kk in range(len(all_masks))])
					while mask_i < len(keep_masks):
						to_remove = [] # only keep tightest bounds
						_, _, inspect = all_masks[keep_masks[mask_i]]
						for mii, (compare_ind) in enumerate(keep_masks):
							if compare_ind == keep_masks[mask_i]: continue
							mask, coords, minmax = all_masks[compare_ind]
							if isTighter(minmax, inspect):
								# a sibling provides a tighter bound, remove this
								to_remove.append(mii)
						keep_masks = np.delete(keep_masks, to_remove)
						mask_i += 1

					for _, coords, _ in all_masks:
						for xx, yy in coords:
							cv2.circle(markedim, (xx, yy), 3, (0, 0, 0))
					scipy.misc.imsave('sample.png', markedim)

					mask = np.zeros(markedim.shape, dtype=np.uint8)
					for mask_ind in keep_masks:
						partmask, coords, _ = all_masks[mask_ind]
						mask = np.logical_or(partmask, mask)
					mask = mask.astype(np.uint8)

					# kii = 0
					# plt.figure(figsize=(14, 10))
					for ii, mask_ind in enumerate(keep_masks):
						partmask, coords, minmax = all_masks[mask_ind]
						xlen = minmax[1] - minmax[0]
						ylen = minmax[3] - minmax[2]
						if xlen > 500 or ylen > 500:
							continue
						patch = local_sample(calibrated, coords)
						cancer_tiles.append(patch)
					# 	plt.subplot(2, 3, ii+1)
					# 	plt.gca().set_title(scaninfo['name'])
					# 	plt.imshow(calibrated, cmap='gray', vmin=0, vmax=1)
					# 	plt.imshow(partmask, alpha=0.2)
					# 	plt.subplot(2, 3, ii+1 + 3)
					# 	if patch is not None:
					# 		# plt.gca().set_title(overlay['pathology'])
					# 		plt.imshow(patch, cmap='gray', vmin=0, vmax=1)
					# plt.show()

					assert mask is not None
				trimmed_mask = trim(mask)
				trimmed_marked = trim(markedim)
				if facing == 'right':
					trimmed_mask = cv2.flip(trimmed_mask, 1)
					trimmed_marked = cv2.flip(trimmed_marked, 1)
				filled_mask = canvas(trimmed_mask)
				filled_marked = canvas(trimmed_marked)
				outmask = filled_mask[y0:yf, x0:xf]

			from tiling import tile_sample
			filled_od = filled_od[y0:yf, x0:xf]
			image_tiles, mask_tiles = tile_sample(filled_od, outmask)
			scale256 = 256 / float(len(filled_od))
			filled_od = cv2.resize(filled_od, (0, 0), fx=scale256, fy=scale256)
			if outmask is not None:
				scale256 = 256 / float(len(outmask))
				outmask = cv2.resize(outmask, (0, 0), fx=scale256, fy=scale256)
			else:
				outmask = np.zeros(filled_od.shape, dtype=np.uint8)

			# phenotype = 'normals'
			# if np.max(outmask) == 1:
			# 	phenotype = 'cancers'

			# with h5py.File('%s/%s/%s.h5' % (DATAROOT, phenotype, scaninfo['name']), 'w') as dbhandle:
			with h5py.File('%s/cancers/%s.h5' % (DATAROOT, scaninfo['name']), 'w') as dbhandle:
				# define data shape
				dbhandle.create_dataset('name', (1,), maxshape=(None,), dtype=h5py.special_dtype(vlen=bytes))
				dbhandle['name'][:] = scaninfo['name']

				dbhandle.create_dataset('image', (256, 256), dtype='float32')
				dbhandle.create_dataset('mask', (256, 256), dtype='uint8')
				dbhandle['image'][:, :] = filled_od
				dbhandle['mask'][:, :] = outmask

				tiles = dbhandle.create_group('tiles')
				tiles.create_dataset('cancers', (len(cancer_tiles), 256, 256), dtype='float32', compression='gzip')
				dbhandle['tiles']['cancers'][:, :, :] = cancer_tiles

				normal_inds = []
				for tileii, (imgtl, masktl) in enumerate(zip(image_tiles, mask_tiles)):
					if np.mean(imgtl) < 0.1:
						continue
					elif np.sum(masktl) == 0:
						normal_inds.append(tileii)
				tiles.create_dataset('normals', (len(normal_inds),), dtype='uint8')
				dbhandle['tiles']['normals'][:] = normal_inds

				tiles.create_dataset('images', (15 ** 2, 256, 256), dtype='float32', compression='gzip')
				dbhandle['tiles']['images'][:, :, :] = image_tiles

		dt = time.time() - t0
		sys.stdout.write('%d: %.2f (eta: %.2f)\r' % (ii, dt, dt * (len(chosen) - ii - 1) / 3600.0))
		sys.stdout.flush()