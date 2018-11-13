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


	def count(self, image):
		
		image_in = self.__preprocess_image(image)

		with chainer.using_config('train', False):
			score = self.__model.forward(image_in)
		
		score = F.softmax(score)
		score = cuda.to_cpu(score.data)[0]
		
		return score


	#def count_on_mosaic(self, mosaic_image, count_ignore_width=8):
	#	return


	def __preprocess_image(self, image):

		image_in = (image - self.__mean) / 255.0
		image_in = image_in.transpose(2, 0, 1)
		image_in = image_in[np.newaxis, :, :, :]
		image_in = Variable(cuda.cupy.asarray(image_in, dtype=cuda.cupy.float32))

		return image_in
