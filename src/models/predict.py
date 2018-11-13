#!/usr/bin/env python

import numpy as np
import cv2
import math

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


	#def count_on_mosaic(self, mosaic_image, count_ignore_width=8):
	#	return


	def __preprocess_image(self, image):

		image_in = (image - self.__mean) / 255.0
		image_in = image_in.transpose(2, 0, 1)
		image_in = image_in[np.newaxis, :, :, :]
		image_in = Variable(cuda.cupy.asarray(image_in, dtype=cuda.cupy.float32))

		return image_in
		

	def __compute_cam(features, weights):

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