#!/usr/bin/env python
import argparse
import os
import numpy as np

import sys
sys.path.append("../models")

from dataset import CowcDataset_Counting


def compute_mean(dataset):
	print('Computing mean image...')

	sum_rgb = np.zeros(shape=[3, ])
	N = len(dataset)
	for i, (image, _) in enumerate(dataset):
		sum_rgb += image.mean(axis=(1, 2), keepdims=False)
		sys.stderr.write('{} / {}\r'.format(i, N))
		sys.stderr.flush()
	sys.stderr.write('\n')
	mean = sum_rgb / N

	print("Done!")
	print("Computed mean: (R, G, B) = ({}, {}, {})".format(mean[0], mean[1], mean[2]))

	return mean


if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='Compute images mean array')
	
	parser.add_argument('--data-list', help='Path to training image-label list file',
						default='../../data/cowc_processed/train_val/crop/train.txt')
	parser.add_argument('--root', help='Root directory path of image files', 
						default='../../data/cowc_processed/train_val/crop/train')
	parser.add_argument('--output', help='Path to output mean array',
						default='../../data/cowc_processed/train_val/crop/mean.npy')
	parser.add_argument('--crop-size', type=int, help='Crop size in px',
						default=96)

	args = parser.parse_args()

	dataset = CowcDataset_Counting(args.data_list, args.root, args.crop_size)
	mean = compute_mean(dataset)

	np.save(args.output, mean)
	