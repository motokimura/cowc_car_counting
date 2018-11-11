#!/usr/bin/env python

from __future__ import print_function

import argparse
import numpy as np

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import cuda
from chainer import training
from chainer.training import extensions

from resnet50 import ResNet50
from dataset import CowcDataset_Counting

from tensorboardX import SummaryWriter
from tboard_logger import TensorboardLogger

import os


def compute_class_weight(histogram, car_max):

	histogram_new = np.empty(shape=[(car_max + 1),])

	histogram_new[:car_max] = histogram[:car_max]
	histogram_new[car_max] = histogram[car_max:].sum()

	class_weight = 1.0 / histogram_new

	class_weight /= class_weight.sum()

	return class_weight


def train_model():
	parser = argparse.ArgumentParser()

	parser.add_argument('--dataset', '-D', default='../../data/cowc_processed/train_val/crop',
						help='Path to directory containing train.txt, val.txt, mean.npy and data directory')
	parser.add_argument('--car-max', '-M', default=40,
						help='Max car number to count')
	parser.add_argument('--no-class-weight', '-w', action='store_true',
						help='Do not use class weight when compute softmax cross entropy loss')
	parser.add_argument('--batchsize', '-b', type=int, default=32,
						help='Number of images in each mini-batch')
	parser.add_argument('--test-batchsize', '-B', type=int, default=250,
						help='Number of images in each test mini-batch')
	parser.add_argument('--gpu', '-g', type=int, default=0,
						help='GPU ID (negative value indicates CPU)')
	parser.add_argument('--epoch', '-e', type=int, default=100,
						help='Number of sweeps over the dataset to train')
	parser.add_argument('--lr-shift', type=int, nargs='*', default=[1.0 / 3.0, 2.0 / 3.0],
						help='Epochs to shift learning rate exponentially by 0.1')
	parser.add_argument('--lr', type=float, default=0.01,
						help='Initial leraning rate used in MomentumSGD optimizer')
	parser.add_argument('--frequency', '-f', type=int, default=1,
						help='Frequency of taking a snapshot')
	parser.add_argument('--out', '-o', default='logs',
						help='Directory to output the result under "models" directory')
	parser.add_argument('--resume', '-r', default='',
						help='Resume the training from snapshot')
	parser.add_argument('--noplot', dest='plot', action='store_false',
						help='Disable PlotReport extension')

	args = parser.parse_args()

	print('GPU: {}'.format(args.gpu))
	print('# Minibatch-size: {}'.format(args.batchsize))
	print('# epoch: {}'.format(args.epoch))
	print('')

	log_dir = os.path.join("../../models", args.out)
	writer = SummaryWriter(log_dir=log_dir)

	# Compute class_weight used in softmax cross entropy
	if args.no_class_weight:
		class_weight = None
	else:
		histogram = np.load(os.path.join(args.dataset, "histogram.npy"))
		class_weight = compute_class_weight(histogram, args.car_max)
		if args.gpu >= 0:
			class_weight = cuda.cupy.asarray(class_weight, dtype=cuda.cupy.float32)
	
	# Set up a neural network to train
	# Classifier reports softmax cross entropy loss and accuracy at every
	# iteration, which will be used by the PrintReport extension below.
	model = ResNet50(args.car_max + 1, class_weight)
	if args.gpu >= 0:
		# Make a specified GPU current
		chainer.cuda.get_device_from_id(args.gpu).use()
		model.to_gpu()  # Copy the model to the GPU

	# Setup an optimizer
	optimizer = chainer.optimizers.MomentumSGD(lr=args.lr, momentum=0.9)
	optimizer.setup(model)
	
	# Load mean image
	mean = np.load(os.path.join(args.dataset, "mean.npy"))
	
	# Load the MNIST dataset
	data_root = os.path.join(args.dataset, "data")

	train = CowcDataset_Counting(os.path.join(args.dataset, "train.txt"), data_root,
								mean=mean, random_flip=True, distort=True, label_max=args.car_max)
	
	test = CowcDataset_Counting(os.path.join(args.dataset, "val.txt"), data_root, 
								mean=mean, random_flip=False, distort=False, label_max=args.car_max)

	train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
	test_iter = chainer.iterators.SerialIterator(test, args.test_batchsize, repeat=False, shuffle=False)

	# Set up a trainer
	updater = training.StandardUpdater(
		train_iter, optimizer, device=args.gpu)
	trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=log_dir)

	# Evaluate the model with the test dataset for each epoch
	trainer.extend(extensions.Evaluator(test_iter, model, device=args.gpu))

	# Dump a computational graph from 'loss' variable at the first iteration
	# The "main" refers to the target link of the "main" optimizer.
	trainer.extend(extensions.dump_graph('main/loss'))

	# Take a snapshot for each specified epoch
	frequency = args.epoch if args.frequency == -1 else max(1, args.frequency)
	trainer.extend(extensions.snapshot(), trigger=(frequency, 'epoch'))
	
	# Save trained model for each specific epoch
	trainer.extend(extensions.snapshot_object(
		model, 'model_iter_{.updater.iteration}'), trigger=(frequency, 'epoch'))

	# Write a log of evaluation statistics for each epoch
	trainer.extend(extensions.LogReport())

	# Decay learning rate at some epochs
	if len(args.lr_shift) > 0:
		lr_shift = [int(shift_point * args.epoch) for shift_point in args.lr_shift]
		trainer.extend(extensions.ExponentialShift('lr', 0.1), 
					   trigger=triggers.ManualScheduleTrigger(lr_shift, 'epoch'))

	# Monitor learning rate at every iteration
	trainer.extend(extensions.observe_lr(), trigger=(1, 'iteration'))

	# Save two plot images to the result dir
	if args.plot and extensions.PlotReport.available():
		trainer.extend(
			extensions.PlotReport(['main/loss', 'validation/main/loss'],
								  'epoch', file_name='loss.png'))
		trainer.extend(
			extensions.PlotReport(
				['main/accuracy', 'validation/main/accuracy'],
				'epoch', file_name='accuracy.png'))

	# Print selected entries of the log to stdout
	# Here "main" refers to the target link of the "main" optimizer again, and
	# "validation" refers to the default name of the Evaluator extension.
	# Entries other than 'epoch' are reported by the Classifier link, called by
	# either the updater or the evaluator.
	trainer.extend(extensions.PrintReport(
		['epoch', 'main/loss', 'validation/main/loss',
		 'main/accuracy', 'validation/main/accuracy', 'lr', 'elapsed_time']))

	# Print a progress bar to stdout
	trainer.extend(extensions.ProgressBar())
	
	# Write training log to TensorBoard log file
	trainer.extend(TensorboardLogger(writer,
		['main/loss', 'validation/main/loss',
		 'main/accuracy', 'validation/main/accuracy',
		 'lr']))
	
	if args.resume:
		# Resume from a snapshot
		chainer.serializers.load_npz(args.resume, trainer)

	# Run the training
	trainer.run()


if __name__ == '__main__':
	train_model()
