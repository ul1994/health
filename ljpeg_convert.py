from __future__ import print_function
from ljpeg import ljpeg
import numpy as np
import json, sys, os

import time
from shutil import copyfile

def DBA_MGH(mat):
	mat[mat < 5] = 5
	mat[mat > 55722] = 55722
	return (np.log10(mat) - 4.80662) / -1.07553

def HOWTEK_MGH(mat):
	return (3.789 - 0.00094568 * mat)

def LUMISYS_WFU(mat):
	return ( mat - 4096.99 ) / -1009.01

def HOWTEK_ISMD(mat):
	return (3.96604095240593 + (-0.00099055807612) * (mat))

def datapath(case, sequence):
	return 'data/%s/%s.png' % (case['diagnosis'], sequence['name'])

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

	chosen = [elem for elem in metadata if elem['diagnosis'] == label]

	print('Parsing:', label, len(chosen))

	plt.ion()
	for ii, elem in enumerate(chosen):
		# if ii < 5: continue
		t0 = time.time()
		try:
			assert len(elem['scans']) == 4
		except:
			print(elem['root'])
			print(elem['scans'])
			raise Exception('Few scans... %d' % (len(elem['scans'])))
		for scaninfo in elem['scans']:
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
			# scipy.misc.imsave('dump.png', (scan / 4095).astype(np.float))
			# print(elem['digitizer'])
			# print('SOURCE:', np.min(scan), np.max(scan))
			hh, ww = scan.shape
			calibrated = None
			exec('calibrated = %s(scan)' % elem['digitizer'].replace('-', '_'))
			calibrated = 3.6 - calibrated
			# print('CALIBB:', np.min(calibrated), np.max(calibrated))
			# scipy.misc.imsave('dump2.png', (calibrated).astype(np.float))
			if np.sum(calibrated > 3.6) > 0:
				print('SOURCE:', np.min(scan), np.max(scan))
				print('MINMAX:', np.min(calibrated), np.max(calibrated))
				print('WARN: CLIPPING %s' % (elem['digitizer']), scanfile)

			calibrated /= 3.6
			calibrated[calibrated < 0] = 0

			grayscale = (255 * calibrated).astype(np.uint8)

			scale1000 = 1000.0 / grayscale.shape[0]
			shrunk = cv2.resize(grayscale, (0,0), fx=scale1000, fy=scale1000)

			savepath = datapath(elem, scaninfo)
			scipy.misc.imsave(savepath, shrunk)
			# break

		dt = time.time() - t0
		sys.stdout.write('%d: %.2f (eta: %.2f)\r' % (ii, dt, dt * (len(chosen) - ii - 1) / 3600.0))
		sys.stdout.flush()
		# break