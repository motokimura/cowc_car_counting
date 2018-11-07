#!/usr/bin/env python
import os

import six
import numpy as np

try:
	from PIL import Image
	available = True
except ImportError as e:
	available = False
	_import_error = e

from chainer.dataset import dataset_mixin


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
	
	def __init__(self, paths, root='.', dtype=np.float32):
		_check_pillow_availability()
		if isinstance(paths, six.string_types):
			with open(paths) as paths_file:
				paths = [path.rstrip() for path in paths_file]
		self._paths = paths
		self._root = root
		self._dtype = dtype

	def __len__(self):
		return len(self._paths)

	def get_example(self, i):
		path = os.path.join(self._root, self._paths[i])
		image_label_pair = _read_image_as_array(path, self._dtype)

		h, w, _ = image_label_pair.shape

		image = image_label_pair[:, :w//2,  :]
		label = image_label_pair[:,  w//2:, :]

		return image.transpose(2, 0, 1)
