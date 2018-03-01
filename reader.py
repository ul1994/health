import cv2
import numpy as np
import matplotlib.pyplot as plt

def DBA_MGH(mat):
	return ( np.log(mat) - 4.80662 ) / -1.07553

def HOWTEK_MGH(mat):
	return 3.789 - 0.00094568 * mat

def LUMISYS_WFU(mat):
	return ( mat - 4096.99 ) / -1009.01

def HOWTEK_ISMD(mat):
	return 3.96604095240593 + (-0.00099055807612) * (mat)

wsize = 2000

def trim_edges(img, off=0.05):
	yy, xx = img.shape
	inv = 1 - off

	x0 = int(xx * off)
	xf = int(xx * inv)
	y0 = int(yy * off)
	yf = int(yy * inv)
	return img.copy()[y0:yf, x0:xf]

def find_boundary(sample):
	boundary = np.mean(sample)
	cross = 0
	for ii in range(0, len(sample)):
		cross = ii
		if sample[ii] > boundary: break
	bufferedcross = int(cross * 0.75)
	return bufferedcross, sample[int(cross * 0.5)]

def create_mask(gray):
	mask = np.zeros(gray.shape)
	test_lines = [gray.shape[0] / 2, gray.shape[0] / 4, 3 * gray.shape[0] / 4]
	boundaries = [find_boundary(gray[line, :]) for line in test_lines]
	crossind = np.argmin([arg[0] for arg in boundaries])

	cross, maskcolor = boundaries[crossind]

	maskmid = (gray.shape[1],gray.shape[0] / 2)
	mask = np.zeros(gray.shape)
	mask[:, int(cross):] = 255
	# mask = cv2.ellipse(mask, maskmid, (gray.shape[1] - int(cross), gray.shape[0]/2), \
	# 				   0, 0, 360, (255,255,255), -1)
	# plt.figure()
	# plt.plot(sample)
	# plt.plot([0, len(sample)], [boundary, boundary])
	# plt.plot([cross, cross], [0, np.max(sample)])
	# plt.show()
	return mask, maskcolor
# import scipy.misc