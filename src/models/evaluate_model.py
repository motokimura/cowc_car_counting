#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import cv2
from skimage import io
from sklearn.metrics import confusion_matrix

from PIL import Image
Image.MAX_IMAGE_PIXELS = 1000000000


def extract_label_pred_vectors(scene_info_list, car_max):

	# Extract ground-truth and predicted value from scene_info
	labels = np.empty(shape=[0,], dtype=int)
	preds = np.empty(shape=[0,], dtype=int)

	for scene_info in scene_info_list:

		cars_labeled = scene_info['cars_labeled'].copy()
		cars_labeled[cars_labeled > car_max] = car_max
		cars_counted = scene_info['cars_counted']

		labels = np.append(labels, cars_labeled.flatten())
		preds = np.append(preds, cars_counted.flatten())

	return labels, preds


def compute_mean_absolute_error(labels, preds):

	N = labels.shape[0]
	absolute_errors = np.abs(preds - labels)
	mean_absolute_error = absolute_errors.sum() / N

	return mean_absolute_error


def compute_root_mean_squared_error(labels, preds):

	N = labels.shape[0]
	squared_errors = (preds - labels) ** 2.0
	mean_squared_error = squared_errors.sum() / N
	root_mean_squared_error = np.sqrt(mean_squared_error)

	return root_mean_squared_error


def compute_accuracy_within_tolerance(confusion, tolerance=0):
	
	class_num = confusion.shape[0]

	# Generate mask array(=[class_num, class_num]) to sample correctly classified tiles from confusion matrix
	ones = np.ones(shape=[class_num,], dtype=bool)
	mask = np.diag(ones)

	for t in range(tolerance):
		k = t + 1
		mask_p = np.diag(ones, k=k)
		mask_m = np.diag(ones, k=-k)
		mask_p = mask_p[:-k, :-k]
		mask_m = mask_m[:-k, :-k]

		mask += (mask_p + mask_m)
	
	# Count correctly classified tiles from confusion matrix based on the mask
	correctly_classified = confusion[mask].sum()
	
	total = confusion.sum()
	accuracy = correctly_classified / total

	return accuracy


def compute_proposal_accuracy(confusion):
	# Compute the accuracy when the counting network is used as a proposal method. 
	# If the network counts zero cars, the region would be proposed to contain no cars.
	# If the network counts at least one car, the region would be proposed to have at least one car.

	correctly_proposed = confusion[0, 0] + confusion[1:, 1:].sum()
	
	total = confusion.sum()
	accuracy = correctly_proposed / total

	return accuracy


def make_evaluation_result_dict(labels, preds, car_max):

	eval_result = {}

	eval_result['car_max'] = car_max

	confusion = confusion_matrix(labels, preds)
	eval_result['confusion'] = confusion

	eval_result['metrics'] = {}
	eval_result['metrics']['mae'] = compute_mean_absolute_error(labels, preds)
	eval_result['metrics']['rmse'] = compute_root_mean_squared_error(labels, preds)
	eval_result['metrics']['accuracy'] = compute_accuracy_within_tolerance(confusion, tolerance=0)
	eval_result['metrics']['accuracy_1'] = compute_accuracy_within_tolerance(confusion, tolerance=1)
	eval_result['metrics']['accuracy_2'] = compute_accuracy_within_tolerance(confusion, tolerance=2)
	eval_result['metrics']['proposal_accuracy'] = compute_proposal_accuracy(confusion)
	
	eval_result['cars'] = {}
	eval_result['cars']['counted'] = preds.sum()
	eval_result['cars']['labeled'] = labels.sum()

	return eval_result


def evaluate_model(
	model, 
	test_scene_list="../../data/cowc_processed/test/test_scenes.txt", 
	data_root="../../data/cowc/datasets/ground_truth_sets", 
	count_ignore_width=8,
	car_max=14):
	
	with open(test_scene_list) as f:
		test_scenes = f.readlines()

	scene_info_list = []

	for idx, scene_name in enumerate(test_scenes):
		scene_name = scene_name.rstrip()

		# Load test scene anyway..
		print("Loading {} ... ({}/{})".format(scene_name, idx + 1, len(test_scenes)))

		image_path = os.path.join(data_root, "{}.png".format(scene_name))
		label_path = os.path.join(data_root, "{}_Annotated_Cars.png".format(scene_name))

		mosaic_image = io.imread(image_path)
		mosaic_image = mosaic_image[:, :, :3] # remove alpha channel

		mosiac_label = io.imread(label_path)
		mosiac_label = mosiac_label[:, :, 3] # use alpha channel

		# Count cars in each tile on the test scene
		cars, grid_size = model.count_on_mosaic(mosaic_image, mosiac_label, count_ignore_width)
		cars_counted, cars_labeled = cars

		# Compute some evaluation metrics from car counring result and append those metrics to a list
		scene_info = {}
		scene_info['scene'] = scene_name
		scene_info['cars_counted'] = cars_counted
		scene_info['cars_labeled'] = cars_labeled
		scene_info['grid_size'] = grid_size

		labels, preds = extract_label_pred_vectors([scene_info], car_max)
		scene_info['eval'] = make_evaluation_result_dict(labels, preds, car_max)
		scene_info_list.append(scene_info)

	# Evaluate the model on all test scenes and save the result as a dictionary
	labels, preds = extract_label_pred_vectors(scene_info_list, car_max)
	eval_result = make_evaluation_result_dict(labels, preds, car_max)

	return eval_result, scene_info_list
