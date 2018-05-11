
import h5py
import numpy as np

dbhandle = h5py.File('/media/ul1994/ssd1tb/phenotype/cancers/B_3135_1.RIGHT_CC.h5')

import matplotlib.pyplot as plt

name = dbhandle['name']

plt.ion()
plt.figure(figsize=(14, 10))
plt.subplot(1, 2, 1)
plt.imshow(dbhandle['image'])
plt.subplot(1, 2, 2)
plt.gca().set_title(list(name))
plt.imshow(dbhandle['mask'])
plt.show()

dii = 1
plt.figure(figsize=(14, 10))
for yy in range(15):
	for xx in range(15):
		if yy % 2 == 0 and xx % 2 == 0:
			plt.subplot(8, 8, dii)
			plt.imshow(dbhandle['tiles']['images'][-225 + (yy*15+xx)], cmap='gray')
			dii += 1
plt.show()
plt.pause(0.01)

plt.figure(figsize=(14, 10))
for ii in range(len(dbhandle['tiles']['cancers'])):
	plt.subplot(1, 4, ii+1)
	plt.imshow(dbhandle['tiles']['cancers'][ii, :, :], cmap='gray', vmin=0, vmax=1)
plt.show()
plt.pause(0.01)

print(dbhandle['tiles']['normals'][:])

input()
