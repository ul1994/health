from __future__ import print_function
from ljpeg import ljpeg
import numpy as np
import json, sys, os

import time
from shutil import copyfile
import cv2

from scipy.stats import multivariate_normal

def canvas(img):
	if img.shape[1] > img.shape[0]:
		img = img[:, :img.shape[0]]
	filled = np.zeros((img.shape[0], img.shape[0]))
	filled[:img.shape[0], :img.shape[1]] = img[:, :img.shape[0]]
	filled[:, img.shape[1]:] = img[img.shape[0]/2, -1]
	return filled

def direction(img):
	hf = len(img[0]) / 2
	leftsample = np.sum(img[:, :hf] < 0.05)
	rightsample = np.sum(img[:, hf:] < 0.05)

	return 'right' if leftsample > rightsample else 'left'

def trim(img, xoff=0.02, yoff=0.1):
	hh, ww = img.shape
	x0 = int(ww * xoff)
	xf = int(ww * (1 - xoff))
	y0 = int(hh * yoff)
	yf = int(hh * (1 - yoff))
	return img[y0:yf, x0:xf]

def applyMark(img, markup, keepmask=None):
	out = img.copy()
	path = json.loads(markup['trace'])
	x, y = markup['start']
	linecolor = (0,255,255)
	cv2.circle(out, (x, y), 3, linecolor, 4)
	cv2.circle(out, (x, y), 20, linecolor, 4)
	minmax = [1000000, 0, 1000000, 0] # xminmax, yminmax
	for dy, dx in path:
		x += dx
		y += dy
		minmax[0] = min(x, minmax[0])
		minmax[1] = max(x, minmax[1])
		minmax[2] = min(y, minmax[2])
		minmax[3] = max(y, minmax[3])
		cv2.circle(out, (x, y), 3, linecolor, 4)
	# ys, xs = zip(*json.loads(markup['trace']))
	xmin, xmax, ymin, ymax = minmax
	xavg, yavg = (xmax + xmin) / 2, (ymax + ymin) / 2
	xrad, yrad = np.abs((xmax - xmin) / 2.0), np.abs((ymax - ymin) / 2.0)
	radius = int(max(xrad, yrad))
	diameter = radius * 2
	# bigdia = diameter * 2
	cv2.circle(out, (int(xavg), int(yavg)), radius, linecolor, 4)
	cv2.circle(out, (int(xavg), int(yavg)), 5, linecolor, 4)

	# print('RADIUS', radius)
	domain1d = np.linspace(0, 5, diameter)
	# kern1d = multivariate_normal.pdf(domain1d, mean=2.5, cov=0.5)
	kmean = 2.5
	kern1d = multivariate_normal.pdf(domain1d, mean=kmean, cov=kmean / 3.0)
	kernel = np.outer(kern1d, kern1d)


	if keepmask is None:
		mask = np.zeros(img.shape)
	else:
		mask = keepmask
	# mask[ymin-radius:ymin+(2 + 1) * radius, xmin-radius:xmin+(2 + 1) * radius] = kernel
	# print(img.shape, kernel.shape, diameter)
	fity, fitx = mask[ymin:ymin+diameter, xmin:xmin+diameter].shape
	mask[ymin:ymin+diameter, xmin:xmin+diameter] = kernel[:fity, :fitx]
	mask *= 1.0 / np.max(mask)

	# plt.figure(figsize=(14, 14))
	# plt.subplot(1, 2, 1)
	# plt.imshow(out, cmap='gray')
	# plt.imshow(mask, alpha=0.25)
	# plt.subplot(1, 2, 2)
	# plt.imshow(kernel)
	# plt.tight_layout()
	# plt.show()
	# try: input()
	# except: pass
	# plt.close()

	return out, mask

def bbox(img, size=2048, lowcut = 255 * 0.05):
	maxy = 0
	ypos = 0
	for yy in range(len(img) - 1, 0, -1):
		miny = 0
		for xx in range(len(img[yy])):
			if img[yy, xx] < lowcut:
				miny = xx
				break
		if maxy < miny:
			maxy = miny
			ypos = yy

	x0 = maxy - size
	if x0 < 0: x0 = 0
	y0 = ypos - size / 2
	if y0 < 0: y0 = 0
	if y0 + size >= len(img): y0 -= (y0 + size - len(img))

	return x0, y0, x0 + size, y0 + size

def DBA_MGH(mat):
	mat[mat < 1] = 1
	# mat[mat > 55722] = 55722
	return (np.log10(mat) - 4.80662) / -1.07553

def HOWTEK_MGH(mat):
	return (3.789 - 0.00094568 * mat)

def LUMISYS_WFU(mat):
	return ( mat - 4096.99 ) / -1009.01

def HOWTEK_ISMD(mat):
	return (3.96604095240593 + (-0.00099055807612) * (mat))

def datapath(case, sequence):
	return 'data/%s/%s.png' % (case['diagnosis'], sequence['name'])

def maskpath(case, sequence):
	return 'data/%s/%s-mask.npy' % (case['diagnosis'], sequence['name'])

def preview_maskpath(case, sequence):
	return 'data/%s/%s-mask.png' % (case['diagnosis'], sequence['name'])

def markedpath(case, sequence):
	return 'data/%s/%s-marked.png' % (case['diagnosis'], sequence['name'])

def thumbpath(case, sequence):
	return 'data/%s/%s-thumb.png' % (case['diagnosis'], sequence['name'])

import matplotlib.pyplot as plt
import scipy.misc
import cv2

if __name__ == '__main__':
	with open('metadata.json') as fl:
		metadata = json.load(fl)

	if len(sys.argv) == 1: raise Exception('Specify label')

	labels = [
		'cancers',
		'benigns',
		'callbacks',
		'normals',
	]

	lid = int(sys.argv[1])
	label = labels[lid]

	chosen = [elem for elem in metadata.values() if elem['diagnosis'] == label]

	print('Parsing:', label, len(chosen))

	plt.ion()
	for ii, elem in enumerate(chosen):
		t0 = time.time()
		try:
			assert len(elem['scans']) == 4
		except:
			print(elem['root'])
			print(elem['scans'])
			raise Exception('Few scans... %d' % (len(elem['scans'])))
		# if elem['name'] != 'A-1088-1':
		# 	continue
		for scaninfo in elem['scans'].values():
			has_details = 'details' in scaninfo
			scanfile = '%s/%s.LJPEG' % (elem['root'], scaninfo['name'])
			tmpfile = 'tmp-%d.LJPEG' % lid
			try:
				copyfile(scanfile, tmpfile)
			except:
				print('Could not find files:', scanfile)
			try:
				scan = ljpeg.read(tmpfile).astype(np.float)
			except:
				print('READ ERROR:', scaninfo['name'], elem['root'])
				continue
			scan = scan.reshape((max(scan.shape), min(scan.shape)))
			# print('RAW   :', np.min(scan), np.max(scan))
			scipy.misc.imsave('sample-raw.png', scan / 4095.0 * 255.0) # dump
			hh, ww = scan.shape
			calibrated = None
			exec('calibrated = %s(scan)' % elem['digitizer'].replace('-', '_'))
			# maxod = 2.7
			# maxod = 1.8
			maxod = 3.6
			calibrated = maxod - calibrated
			# print(np.min(calibrated), np.max(calibrated))

			# cliphi = 0.8 # push 40 lumens worth oob
			# cliphi = 1. # push 40 lumens worth oob
			# calibrated *= cliphi

			cliplow = maxod / 4.0
			calibrated -= cliplow
			calibrated /= (maxod - cliplow)
			calibrated[calibrated < 0] = 0
			calibrated[calibrated > 1] = 1
			grayscale = (255 * calibrated).astype(np.uint8)

			visible = calibrated - 0.3333
			visible /= 0.66666
			visible[visible < 0] = 0
			visible[visible > 1] = 1
			visible = (255 * visible).astype(np.uint8)

			# grayscale = (255 * calibrated).astype(np.uint8)

			# scipy.misc.imsave('sample-gray.png', grayscale) # dump
			# cleanup
			trimmed = trim(grayscale)
			facing = direction(trimmed)
			trimmed_visible = trim(visible)
			if facing == 'right':
				trimmed = cv2.flip(trimmed, 1)
				trimmed_visible = cv2.flip(trimmed_visible, 1)
			filled = canvas(trimmed)
			filled_visible = canvas(trimmed_visible)
			x0, y0, xf, yf = bbox(filled)
			boxedim = filled_visible[y0:yf, x0:xf]
			scipy.misc.imsave(datapath(elem, scaninfo), boxedim)
			xsmall = 512 / float(filled.shape[1])
			ysmall = 512 / float(filled.shape[0])
			cropzone = filled_visible.copy()
			cv2.rectangle(cropzone, (x0,y0), (xf, yf), 0, 10)
			scipy.misc.imsave(thumbpath(elem, scaninfo), cv2.resize(cropzone, (0,0), fx=xsmall, fy=ysmall))

			if has_details:
				# print('MARKS:', len(scaninfo['details']['markups']))
				mask = None
				markedim = grayscale.copy()
				# try:
				for markup in scaninfo['details']['markups']:
					markedim, mask = applyMark(markedim, markup, keepmask=mask)
				# except:
				# 	print('Markups:', elem['name'], scaninfo['name'])
				# 	raise Exception('Failed to create heatmap')
				trimmed_mask = trim(mask)
				trimmed_marked = trim(markedim)
				if facing == 'right':
					trimmed_mask = cv2.flip(trimmed_mask, 1)
					trimmed_marked = cv2.flip(trimmed_marked, 1)
				filled_mask = canvas(trimmed_mask)
				filled_marked = canvas(trimmed_marked)
				scipy.misc.imsave(preview_maskpath(elem, scaninfo), 255.0 * filled_mask[y0:yf, x0:xf])
				np.save(maskpath(elem, scaninfo), filled_mask[y0:yf, x0:xf])
				scipy.misc.imsave(markedpath(elem, scaninfo), 255.0 * filled_marked[y0:yf, x0:xf])

				if len(scaninfo['details']['markups']) > 1:
					print()
					print('Markups:', elem['name'], scaninfo['name'], len(scaninfo['details']['markups']))
					# try: input()
					# except: pass
		# raise Exception('BREAKPOINT')

		dt = time.time() - t0
		sys.stdout.write('%d: %.2f (eta: %.2f)\r' % (ii, dt, dt * (len(chosen) - ii - 1) / 3600.0))
		sys.stdout.flush()