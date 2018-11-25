# Original author: yasunorikudo
# (https://github.com/yasunorikudo/chainer-ResNet)

import chainer
import chainer.functions as F
from chainer import initializers
import chainer.links as L


class BottleNeckA(chainer.Chain):

	def __init__(self, in_size, ch, out_size, stride=2):
		super(BottleNeckA, self).__init__()
		initialW = initializers.HeNormal()

		with self.init_scope():
			self.conv1 = L.Convolution2D(
				in_size, ch, 1, stride, 0, initialW=initialW, nobias=True)
			self.bn1 = L.BatchNormalization(ch)
			self.conv2 = L.Convolution2D(
				ch, ch, 3, 1, 1, initialW=initialW, nobias=True)
			self.bn2 = L.BatchNormalization(ch)
			self.conv3 = L.Convolution2D(
				ch, out_size, 1, 1, 0, initialW=initialW, nobias=True)
			self.bn3 = L.BatchNormalization(out_size)

			self.conv4 = L.Convolution2D(
				in_size, out_size, 1, stride, 0,
				initialW=initialW, nobias=True)
			self.bn4 = L.BatchNormalization(out_size)

	def __call__(self, x):
		h1 = F.relu(self.bn1(self.conv1(x)))
		h1 = F.relu(self.bn2(self.conv2(h1)))
		h1 = self.bn3(self.conv3(h1))
		h2 = self.bn4(self.conv4(x))

		return F.relu(h1 + h2)


class BottleNeckB(chainer.Chain):

	def __init__(self, in_size, ch):
		super(BottleNeckB, self).__init__()
		initialW = initializers.HeNormal()

		with self.init_scope():
			self.conv1 = L.Convolution2D(
				in_size, ch, 1, 1, 0, initialW=initialW, nobias=True)
			self.bn1 = L.BatchNormalization(ch)
			self.conv2 = L.Convolution2D(
				ch, ch, 3, 1, 1, initialW=initialW, nobias=True)
			self.bn2 = L.BatchNormalization(ch)
			self.conv3 = L.Convolution2D(
				ch, in_size, 1, 1, 0, initialW=initialW, nobias=True)
			self.bn3 = L.BatchNormalization(in_size)

	def __call__(self, x):
		h = F.relu(self.bn1(self.conv1(x)))
		h = F.relu(self.bn2(self.conv2(h)))
		h = self.bn3(self.conv3(h))

		return F.relu(h + x)


class Block(chainer.ChainList):

	def __init__(self, layer, in_size, ch, out_size, stride=2):
		super(Block, self).__init__()
		self.add_link(BottleNeckA(in_size, ch, out_size, stride))
		for i in range(layer - 1):
			self.add_link(BottleNeckB(out_size, ch))
		
		self._layer = layer

	def __call__(self, x):
		for f in self.children():
			x = f(x)
		return x
	
	@property
	def layer(self):
		return self._layer


class ResNet50(chainer.Chain):

	def __init__(self, class_num, insize, class_weight=None, caffemodel_path=None):
		assert (insize % 32 == 0), "'insize' should be divisible by 32."
		
		super(ResNet50, self).__init__()
		with self.init_scope():
			self.conv1 = L.Convolution2D(
				3, 64, 7, 2, 3, initialW=initializers.HeNormal())
			self.bn1 = L.BatchNormalization(64)
			self.res2 = Block(3, 64, 64, 256, 1)
			self.res3 = Block(4, 256, 128, 512)
			self.res4 = Block(6, 512, 256, 1024)
			self.res5 = Block(3, 1024, 512, 2048)
			self.fc = L.Linear(2048, class_num)
		
		if caffemodel_path is not None:
			# Load pre-trained weights from caffemodel
			self._load_pretrained_weights(caffemodel_path)

		self._class_num = class_num
		self._insize = insize
		self._class_weight = class_weight

	def forward(self, x, compute_cam=False):
		h = self.bn1(self.conv1(x))
		h = F.max_pooling_2d(F.relu(h), 3, stride=2)
		h = self.res2(h)
		h = self.res3(h)
		h = self.res4(h)
		h = self.res5(h)
		cam_features = h.data
		h = F.average_pooling_2d(h, self._insize//32, stride=1)
		h = self.fc(h)

		if compute_cam:
			cam_weights = self.fc.W.data
			return h, cam_features, cam_weights

		return h

	def __call__(self, x, t):
		h = self.forward(x)

		loss = F.softmax_cross_entropy(h, t, class_weight=self._class_weight)
		chainer.report({'loss': loss, 'accuracy': F.accuracy(h, t)}, self)
		return loss

	@property
	def insize(self):
		return self._insize

	@property
	def class_num(self):
		return self._class_num
	

	# Functions to load weights from pre-trained ResNet50 caffemodel
	# Reference: https://github.com/chainer/chainer/blob/master/chainer/links/model/vision/resnet.py
	def _load_weights_conv_bn(self, src, dst_conv, dst_bn, bname, cname):
		src_conv = getattr(src, 'res{}_branch{}'.format(bname, cname))
		src_bn = getattr(src, 'bn{}_branch{}'.format(bname, cname))
		src_scale = getattr(src, 'scale{}_branch{}'.format(bname, cname))
		dst_conv.W.data[:] = src_conv.W.data
		dst_bn.avg_mean[:] = src_bn.avg_mean
		dst_bn.avg_var[:] = src_bn.avg_var
		dst_bn.gamma.data[:] = src_scale.W.data
		dst_bn.beta.data[:] = src_scale.bias.b.data

	def _load_weights_bottleneckA(self, dst, src, name):
		self._load_weights_conv_bn(src, dst.conv1, dst.bn1, name, '2a')
		self._load_weights_conv_bn(src, dst.conv2, dst.bn2, name, '2b')
		self._load_weights_conv_bn(src, dst.conv3, dst.bn3, name, '2c')
		self._load_weights_conv_bn(src, dst.conv4, dst.bn4, name, '1')
	
	def _load_weights_bottleneckB(self, dst, src, name):
		self._load_weights_conv_bn(src, dst.conv1, dst.bn1, name, '2a')
		self._load_weights_conv_bn(src, dst.conv2, dst.bn2, name, '2b')
		self._load_weights_conv_bn(src, dst.conv3, dst.bn3, name, '2c')

	def _load_weights_block(self, dst, src, names):
		for i, (layers, name) in enumerate(zip(dst.children(), names)):
			if i ==0:
				self._load_weights_bottleneckA(layers, src, name)
			else:
				self._load_weights_bottleneckB(layers, src, name)

	def _load_pretrained_weights(self, caffemodel_path):
		# As CaffeFunction uses shortcut symbols,
        # CaffeFunction is imported here.
		from chainer.links.caffe.caffe_function import CaffeFunction
		src = CaffeFunction(caffemodel_path)

		self.conv1.W.data[:] = src.conv1.W.data
		self.conv1.b.data[:] = src.conv1.b.data
		self.bn1.avg_mean[:] = src.bn_conv1.avg_mean
		self.bn1.avg_var[:] = src.bn_conv1.avg_var
		self.bn1.gamma.data[:] = src.scale_conv1.W.data
		self.bn1.beta.data[:] = src.scale_conv1.bias.b.data

		self._load_weights_block(self.res2, src, ['2a', '2b', '2c'])
		self._load_weights_block(self.res3, src, ['3a', '3b', '3c', '3d'])
		self._load_weights_block(self.res4, src, ['4a', '4b', '4c', '4d', '4e', '4f'])
		self._load_weights_block(self.res5, src, ['5a', '5b', '5c'])
