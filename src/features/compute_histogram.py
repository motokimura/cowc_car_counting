#!/usr/bin/env python
import argparse
import os
import numpy as np
from tqdm import tqdm

import sys
sys.path.append("../models")

from dataset import CowcDataset_Counting


def compute_histogram(dataset):
    
    hist = np.zeros(shape=[10**3, ], dtype=int)
    
    for image, label in tqdm(dataset):
        
        hist[label] += 1
    
    car_count_max = np.where(hist > 0)[0][-1]
    
    return hist[:car_count_max + 1]


if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='Compute images mean array')
	
	parser.add_argument('--data-list', help='Path to training image-label list file',
						default='../../data/cowc_processed/train_val/crop/train.txt')
	parser.add_argument('--root', help='Root directory path of image files', 
						default='../../data/cowc_processed/train_val/crop/train')
	parser.add_argument('--output', help='path to output distriba array',
						default='../../data/cowc_processed/train_val/crop/histogram.npy')
	parser.add_argument('--crop-size', type=int, help='Crop size in px',
						default=96)

	args = parser.parse_args()

	dataset = CowcDataset_Counting(args.data_list, args.root, args.crop_size)
	hist = compute_histogram(dataset)

	print("Computed histogram:")
	print("car_num, count")
	for car_num, count in enumerate(hist):
		print("{}, {}".format(car_num, count))
		
	np.save(args.output, hist)
