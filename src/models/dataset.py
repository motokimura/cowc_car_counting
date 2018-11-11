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

from transforms import random_color_distort


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
			self, paths, root, 
			dtype=np.float32, label_dtype=np.int32, count_ignore_width=8, mean=None, random_flip=False, distort=False, label_max=10*8):
		_check_pillow_availability()
		if isinstance(paths, six.string_types):
			with open(paths) as paths_file:
				paths = [path.rstrip() for path in paths_file]
		self._paths = paths
		self._root = root
		self._dtype = dtype
		self._label_dtype = label_dtype

		self._count_ignore_width = count_ignore_width

		self._normalize = False if (mean is None) else True
		if self._normalize:
			self._mean = mean[np.newaxis, np.newaxis, :]

		self._random_flip = random_flip
		self._distort = distort

		self._label_max = label_max

	def __len__(self):
		return len(self._paths)

	def get_example(self, i):
		path = os.path.join(self._root, self._paths[i])
		image_mask_pair = _read_image_as_array(path, np.float64)

		h, w, _ = image_mask_pair.shape

		image = image_mask_pair[:, :w//2,  :]
		mask = image_mask_pair[:,  w//2:, 0]

		if self._distort:
			# Apply random color distort
			image = random_color_distort(image)
			image = np.asarray(image, dtype=np.float64)

		if self._normalize:
			# Normalize if mean array is given
			image = (image - self._mean) / 255.0

		if self._random_flip:
			# Horizontal flip
			if random.randint(0, 1):
				image = image[:, ::-1, :]
				mask = mask[:, ::-1]

			# Vertical flip
			if random.randint(0, 1):
				image = image[::-1, :, :]
				mask = mask[::-1, :]

		# Remove car annotation outside the valid area
		count_ignore_width = self._count_ignore_width
		mask[:count_ignore_width, :] = 0
		mask[:, :count_ignore_width] = 0
		mask[-count_ignore_width:, :] = 0
		mask[:, -count_ignore_width:] = 0

		label = (mask > 0).sum()

		# Clipping based on given max value of label
		if label > self._label_max:
			label = self._label_max

		# Type casting
		image = image.astype(self._dtype) 
		label = self._label_dtype(label)

		return image.transpose(2, 0, 1), label
