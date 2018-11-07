#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import random


def dump_filenames(filenames, dst_path):

	with open(dst_path, 'w') as f:
		
		for i, filename in enumerate(filenames):
			filename = filename.rstrip()

			if i != 0:
				f.write("\n")

			f.write(filename)


def split_train_val(data_list, dst_dir, ratio, seed=0):

	with open(data_list) as f:
		crop_filenames = f.readlines()

	random.seed(seed)
	random.shuffle(crop_filenames)

	crop_count = len(crop_filenames)

	train_ratio, val_ratio = ratio
	total = train_ratio + val_ratio

	train_count= int(float(crop_count * train_ratio) / float(total))

	train_crops = crop_filenames[:train_count]
	val_crops = crop_filenames[train_count:]

	dump_filenames(train_crops, os.path.join(dst_dir, "train.txt"))
	dump_filenames(val_crops, os.path.join(dst_dir, "val.txt"))


if __name__ == "__main__":

	parser = argparse.ArgumentParser()

	parser.add_argument('--data_list', help='Text file listing up train/val crops',
						default='../../data/cowc_processed/train_val/crop/train_val.txt')
	parser.add_argument('--dst_dir', help='Root directory to output train.txt and val.txt',
						default='../../data/cowc_processed/train_val/crop/')
	parser.add_argument('--ratio', help='Split ratio for train/val set',
						type=int, nargs=2, default=[8, 2])
	parser.add_argument('--seed', help='random seed',
						type=int, default=0)

	args = parser.parse_args()

	split_train_val(args.data_list, args.dst_dir, args.ratio, args.seed)