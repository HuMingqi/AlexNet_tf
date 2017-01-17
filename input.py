'''
Author: hiocde
Email: hiocde@gmail.com
Date: 1.16.17
original/new: I found the fourth reading data method! not placeholder,quque-pipeline and constant
described by official docs, I used API tf.pack to build myself auto input pipeline!
For another, I used completely tf ops for image not using opencv.
Domain: main_dir{sub_dir{same class raw images}...}
'''

import os
import numpy as np
import tensorflow as tf

class ImageSet:

	def __init__(self, img_main_dir):
		self.examples= self.create_examples(img_main_dir)
		np.random.shuffle(self.examples)
		self.num_exps= len(self.examples)
		self.pointer= 0 	#pointer points next example
		self.images, self.labels= zip(*self.examples)
		self.num_class= len(set(self.labels))

	def has_next_exp(self):
		return self.pointer< self.num_exps

	def next_exp(self):
		if not self.has_next_exp():
			np.random.shuffle(self.examples)
			self.pointer=0
			self.images, self.labels= zip(*self.examples)

		label= self.labels[self.pointer]
		image= self.img_read(self.images[self.pointer])	#3d-tensor
		distorted_image= self.distort_image(image)

		self.pointer+=1
		return distorted_image, label

	def next_batch(self, batch_size):
		'''return images tuple and labels tuple as a batch for tf.placeholder'''
		batch=[]
		for i in range(batch_size):
			exp= self.next_exp()
			batch.append(exp)
		images,labels= zip(*batch)
		return tf.pack(images),tf.pack(labels)	#pack to get a 4d-tensor input for inference and 1d-tensor labels for loss

	def create_examples(self, img_main_dir):
		'''Args:
			img_main_dir: includes sub_dirs, each sub_dir is a class of images
		   Return:
		   	all images path and their labels(0-n-1),a list of tuple
		'''
		examples=[]
		for sub_dir in os.listdir(img_main_dir):
			class_index= int(sub_dir[-1]) 	#because I appended class index to sub_dir
			for img_name in os.listdir(os.path.join(img_main_dir, sub_dir)):
				examples.append((os.path.join(img_main_dir,sub_dir,img_name), class_index))
		return examples

	def img_read(self, img_path):
		'''Brief:
			Directly use tf's op to read and change img not opencv
		   Return:
		   	a 3d-tensor of img, dtype=uint8
		'''
		return tf.image.decode_jpeg(tf.read_file(img_path),3)

	def distort_image(self, img):
		'''imitate tf's cifar10 sample'''
		distorted_image = tf.cast(img, tf.float32)
		#It's fun that 224*224 input size in Alex's paper, but in fact is 227*227.( (224 - 11)/4 + 1 is quite clearly not an integer)
		distorted_image = tf.image.resize_images(distorted_image, [227, 227]) 	
		distorted_image = tf.image.random_flip_left_right(distorted_image)
		distorted_image = tf.image.random_brightness(distorted_image,max_delta=63)
		distorted_image = tf.image.random_contrast(distorted_image,lower=0.2,upper=1.8)
		return tf.image.per_image_standardization(distorted_image)

###test
# trainset= ImageSet('/mnt/g/machine_learning/dataset/Alexnet_tf/paris')
# x,y=trainset.next_batch(32)
# print(x)
# print(y)
# #print:
# # Tensor("pack_32:0", shape=(32, 227, 227, 3), dtype=float32)
# # Tensor("pack_33:0", shape=(32,), dtype=int32)
# #good job! A auto input pipeline