#!/usr/bin/env python
import os

import six
import numpy as np
import random

try:
	from PIL import Image
	available = True
except ImportError as e:
	available = False
	_import_error = e

from chainer.dataset import dataset_mixin

from transforms import random_color_distort as color_distort


def _check_pillow_availability():
	if not available:
		raise ImportError('PIL cannot be loaded. Install Pillow!\n'
						  'The actual import error is as follows:\n' +
						  str(_import_error))


def _read_image_as_array(path, dtype):
	f = Image.open(path)
	try:
		image = np.asarray(f, dtype=dtype)
	finally:
		# Only pillow >= 3.0 has 'close' method
		if hasattr(f, 'close'):
			f.close()
	return image


class CowcDataset_Counting(dataset_mixin.DatasetMixin):
	
	def __init__(
			self, paths, root, crop_size,
			dtype=np.float32, label_dtype=np.int32, mean=None, transpose_image=True, return_mask=False,
			count_ignore_width=8, label_max=10*8, random_crop=False, random_flip=False, random_color_distort=False):
		_check_pillow_availability()
		if isinstance(paths, six.string_types):
			with open(paths) as paths_file:
				paths = [path.rstrip() for path in paths_file]
		self._paths = paths
		self._root = root
		self._crop_size = crop_size
		self._dtype = dtype
		self._label_dtype = label_dtype

		self._normalize = False if (mean is None) else True
		if self._normalize:
			self._mean = mean[np.newaxis, np.newaxis, :]

		self._transpose_image = transpose_image
		self._return_mask = return_mask
		self._count_ignore_width = count_ignore_width
		self._label_max = label_max
		self._random_crop = random_crop
		self._random_flip = random_flip
		self._random_color_distort = random_color_distort

	def __len__(self):
		return len(self._paths)

	def get_example(self, i):
		path = os.path.join(self._root, self._paths[i])
		image_mask_pair = _read_image_as_array(path, np.float64)

		_, W, _ = image_mask_pair.shape

		image = image_mask_pair[:, :W//2,  :]
		mask = image_mask_pair[:,  W//2:, 0]

		# Crop image and mask
		h, w, _ = image.shape

		if self._random_crop:
			# Random crop
			top  = random.randint(0, h - self._crop_size)
			left = random.randint(0, w - self._crop_size)
		else:
			# Center crop
			top  = (h - self._crop_size) // 2
			left = (w - self._crop_size) // 2
		
		bottom = top + self._crop_size
		right = left + self._crop_size

		image = image[top:bottom, left:right]
		mask = mask[top:bottom, left:right]

		if self._random_flip:
			# Horizontal flip
			if random.randint(0, 1):
				image = image[:, ::-1, :]
				mask = mask[:, ::-1]

			# Vertical flip
			if random.randint(0, 1):
				image = image[::-1, :, :]
				mask = mask[::-1, :]
		
		if self._random_color_distort:
			# Apply random color distort
			image = color_distort(image)
			image = np.asarray(image, dtype=np.float64)
		
		if self._normalize:
			# Normalize if mean array is given
			image = (image - self._mean) / 255.0

		# Remove car annotation outside the valid area
		ignore = self._count_ignore_width
		label = (mask[ignore:-ignore, ignore:-ignore] > 0).sum()

		# Clipping based on given max value of label
		if label > self._label_max:
			label = self._label_max

		# Type casting
		image = image.astype(self._dtype) 
		label = self._label_dtype(label)

		# Transpose image from [h, w, c] to [c, h, w]
		if self._transpose_image:
			image = image.transpose(2, 0, 1)

		if self._return_mask:
			return image, label, mask
		else:
			return image, label