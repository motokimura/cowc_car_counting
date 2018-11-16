#!/usr/bin/env python

import sys
import numpy as np
import cv2
import math
from skimage import io

import chainer
import chainer.functions as F
from chainer import cuda, serializers, Variable

from resnet50 import ResNet50

class CarCountingModel:

	def __init__(self, model_path, class_num, insize, mean, gpu=0):
		
		# Load model
		self.__model = ResNet50(class_num, insize)
		serializers.load_npz(model_path, self.__model)

		chainer.cuda.get_device(gpu).use()
		self.__model.to_gpu(gpu)

		# Add height and width dimensions to mean 
		self.__mean = mean[np.newaxis, np.newaxis, :]


	def count(self, image, compute_cam=False):
		
		image_in = self.__preprocess_image(image)

		with chainer.using_config('train', False):
			ret = self.__model.forward(image_in, compute_cam)

		if compute_cam:
			score, features, weights = ret
			cam = self.__compute_cam(features, weights)
		else:
			score = ret
		
		score = F.softmax(score)
		score = cuda.to_cpu(score.data)[0]
		
		if compute_cam:
			return score, cam
		else:
			return score


	def count_on_mosaic(self, mosaic_image, mosaic_label=None, count_ignore_width=8):

		# Create padded mosaic_image and mosaic_label so that their size = n * grid_size + 2 * count_ignore_width
		# s.t., grid_size = model_input_size - 2 * count_ignore_width
		ignore_w = count_ignore_width

		h, w, _ = mosaic_image.shape

		model_insize = self.__model.insize
		grid_size = model_insize - 2 * ignore_w

		yi_max, xi_max = int(math.ceil(h / grid_size)), int(math.ceil(w / grid_size))

		h_grid, w_grid = yi_max * grid_size, xi_max * grid_size
		h_pad, w_pad = h_grid + 2 * ignore_w, w_grid + 2 * ignore_w

		mosaic_image_pad = 127 * np.ones(shape=[h_pad, w_pad, 3], dtype=np.uint8)
		mosaic_image_pad[ignore_w:ignore_w+h, ignore_w:ignore_w+w] = mosaic_image

		if mosaic_label is not None:
			mosaic_label_pad = np.zeros(shape=[h_pad, w_pad], dtype=np.uint8)
			mosaic_label_pad[ignore_w:ignore_w+h, ignore_w:ignore_w+w] = mosaic_label


		# Count cars in each tile on the grid
		car_max = self.__model.class_num - 1
		count_results = []
		tile_idx = 0

		for yi in range(yi_max):
			for xi in range(xi_max):

				top = yi * grid_size
				left = xi * grid_size

				tile_image = mosaic_image_pad[top:top+model_insize, left:left+model_insize]
				score = self.count(tile_image, compute_cam=False)
				pred = np.argmax(score)

				if mosaic_label is not None:
					tile_label = mosaic_label_pad[top:top+model_insize, left:left+model_insize]

					label_original = (tile_label[ignore_w:-ignore_w, ignore_w:-ignore_w] > 0).sum()
					
					label = label_original
					if label > car_max:
						label = car_max

				else:
					label_original = -1000 	# value for N/A
					label = -1000 			# value for N/A

				count_result = {
					'position': {
						'top': top,
						'left': left,
						'bottom': top + grid_size,
						'right': left + grid_size
					},
					'cars': {
						'counted': pred,
						'labeled': label,
						'labeled_original': label_original
					}
				}

				count_results.append(count_result)

				tile_idx += 1
				sys.stderr.write('{} / {}\r'.format(tile_idx, xi_max * yi_max))
				sys.stderr.flush()

		sys.stderr.write('\n')

		return count_results


	def __preprocess_image(self, image):

		image_in = (image - self.__mean) / 255.0
		image_in = image_in.transpose(2, 0, 1)
		image_in = image_in[np.newaxis, :, :, :]
		image_in = Variable(cuda.cupy.asarray(image_in, dtype=cuda.cupy.float32))

		return image_in


	def __compute_cam(self, features, weights):

		features = cuda.to_cpu(features)[0] # [2048=c, insize//32=h, insize//32=w], assuming input batchsize is 1
		weights = cuda.to_cpu(weights)      # [class_num, 2048=c]

		c, h, w = features.shape
		class_num, _ = weights.shape

		cam = np.zeros(shape=[class_num, h, w]) # [class_num, h, w]

		for class_idx in range(class_num):
			weight = weights[class_idx] 	# [c,]
			weight = weight[:, None, None]	# [c, 1, 1]

			cam[class_idx] = np.sum(weight * features, axis=0) # [h, w], (weight * features: [c, h, w])

		return cam