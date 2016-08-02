# -*- coding: utf-8 -*-

'''
README  - commands to run code in an iPython terminal

include vgg16_weights.h5 in the directory with this script

run dd_generator_final.py 
y = run_cross_validation(26)

'''


import numpy as np
np.random.seed(2016)

import os
import glob
import cv2
import math
import pickle
import datetime
import pandas as pd
#import statistics
import time
from shutil import copy2
import warnings
import random
import h5py

import Image
from multiprocessing import Pool
import itertools

warnings.filterwarnings("ignore")

from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import KFold
from keras.models import Sequential, Model
from keras.layers import Input, merge
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU, LeakyReLU
from keras.optimizers import SGD
from keras.optimizers import Adagrad, Adadelta, Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import np_utils
from keras.models import model_from_json
from keras.regularizers import l1l2, activity_l2, l2

from customlayers import convolution2Dgroup, crosschannelnormalization, splittensor, Softmax4D

from sklearn.metrics import log_loss
from scipy.misc import imread, imresize, imshow

use_cache = 1


img_cols, img_rows	= 300, 224


def show_image(im, name='image'):
	cv2.imshow(name, im)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

# color_type = 1 - gray
# color_type = 3 - RGB
def get_im_cv2(path, img_rows, img_cols, color_type=1):
	# Load as grayscale
	if color_type == 1:
		img = cv2.imread(path, 0)
	elif color_type == 3:
		img = cv2.imread(path)
	# Reduce size
	resized = cv2.resize(img, (img_cols, img_rows) ) #, cv2.INTER_LINEAR
	return resized


def get_im_cv2_mod(path, img_rows, img_cols, color_type=1):
	#print(path)
	# Load as grayscale
	if color_type == 1:
		img = cv2.imread(path, 0)
	else:
		img = cv2.imread(path)
	# Reduce size
	rotate = random.uniform(-10, 10)
	M = cv2.getRotationMatrix2D((img.shape[1]/2, img.shape[0]/2), rotate, 1)
	img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
	resized = cv2.resize(img, (img_cols, img_rows) ) #, cv2.INTER_LINEAR
	return resized


#dr, clss = get_driver_data()
def get_driver_data():
	dr = dict()
	clss = dict()
	#path = os.path.join('..', 'input', 'driver_imgs_list.csv')
	path = 'driver_imgs_list.csv'
	print('Read driver data')
	f = open(path, 'r')
	line = f.readline()
	while (1):
		line = f.readline()
		if line == '':
			break
		arr = line.strip().split(',')
		dr[arr[2]] = arr[0]
		if arr[0] not in clss.keys():
			clss[arr[0]] = [(arr[1], arr[2])]
		else:
			clss[arr[0]].append((arr[1], arr[2]))
	f.close()
	return dr, clss





def preprocess(x, augment):

	
	#augment = True
		
	#img_cols, img_rows	 = 128, 96	
	#img_cols, img_rows	 = 192, 144 
	#img_cols, img_rows	= 216, 162
	#img_cols, img_rows	 = 160, 120

	#img_cols, img_rows	 = 224, 168 
	#img_cols, img_rows	 = 172, 124
	#img_cols, img_rows	= 227, 227
	#img_cols, img_rows	= 252, 189
	#img_cols, img_rows	= 256, 192
	#img_cols, img_rows	= 288, 216
	#img_cols, img_rows	= 300, 225
	#img_cols, img_rows	= 384, 288
	#img_cols, img_rows	= 320, 240
	#img_cols, img_rows	= 180, 135
	#img_cols, img_rows	= 200, 150
	#img_cols, img_rows	 = 224, 168
	#img_cols, img_rows	 = 112, 84
	#img_cols, img_rows	 = 432, 324

	

	

	if augment:

		
		a = np.random.rand()
		#brightness
		if  a > 0.4:
			img_yuv = cv2.cvtColor(x, cv2.COLOR_BGR2YUV)

			# equalize the histogram of the Y channel
			img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
			img_yuv[:,:,0] = 255 * (img_yuv[:,:,0]/255.0) ** (0.85 + 0.2*np.random.rand())
			#img_yuv[:,:,1] = cv2.equalizeHist(img_yuv[:,:,1])
			#img_yuv[:,:,2] = cv2.equalizeHist(img_yuv[:,:,2])
	

			# convert the YUV image back to RGB format
			x = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
		
	
		
		a = np.random.rand()
		#shift
		if  a > 0.4:
			hor_shift = random.uniform(-30, 10)
			vert_shift = random.uniform(-10, 10)

			M = np.float32([[1,0,hor_shift],[0,1,vert_shift]])
			x = cv2.warpAffine(x,M,(x.shape[1], x.shape[0]))
 		
 		'''
 		#zoom
 		if a < 0.2:
 			
 			s = 0.8 + 0.2*np.random.rand()
 			M = np.float32([[s,0,0],[0,s,0]])
			x = cv2.warpAffine(x,M,(x.shape[1], x.shape[0]))		
		'''
 		
	
		if np.random.rand() >= 0.8:
			x = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
			x = cv2.cvtColor(x, cv2.COLOR_GRAY2RGB)
		
	
		if np.random.rand() >= 0.6:
			rotate = random.uniform(-10, 10)
			M = cv2.getRotationMatrix2D((x.shape[1]/2, x.shape[0]/2), rotate, 1)
			x = cv2.warpAffine(x, M, (x.shape[1], x.shape[0]))
		
		
		
		else:	

			s = 30
			xl = np.random.randint(0,s)
			xr = np.random.randint(x.shape[1]-s, x.shape[1])
			yt = np.random.randint(0,s)
			yb = np.random.randint(x.shape[0]-s, x.shape[0])
			rect = np.array([[xl,yt],[xr, yt],[xr, yb], [xl, yb]], dtype = "float32")
			#rect = np.array([[yt, xl],[yt, xr],[yb, xl], [yb, xr]], dtype = "float32")
		
			dst = np.array([
				[0, 0],
				[x.shape[1] - 0, 0],
				[x.shape[1] - 0, x.shape[0] - 0],
				[0, x.shape[0] - 0]], dtype = "float32")	
		
			M = cv2.getPerspectiveTransform(rect, dst)
			x = cv2.warpPerspective(x, M, (x.shape[1], x.shape[0]))
	
		x = cv2.resize(x, (img_cols, img_rows), interpolation=cv2.INTER_CUBIC).astype(np.float)	
		
		
		#x = 255.0*((x-np.min(x))/(np.max(x)-np.min(x)))**(0.85 + 0.3*np.random.random())
		#x = 255.0*(x/255.0)**(0.9 + 0.2*np.random.random())
		
		'''
		#x = 1.0*cv2.resize(x[10:-10, 10:-10], (img_cols, img_rows) ) #, cv2.INTER_LINEAR
		#x = 1.0*cv2.resize(x[10:-10, (x.shape[1]- x.shape[0]+10):-10], (img_cols, img_rows) ) #, cv2.INTER_LINEAR
	
		i = np.random.randint(0,20)
		j = np.random.randint(0,20)
		x = 1.0*cv2.resize(x[i:x.shape[0]-20+i, j:x.shape[1]-20+j], (img_cols, img_rows) ) #, cv2.INTER_LINEAR
	
		'''

		
	
	else:

		
		img_yuv = cv2.cvtColor(x, cv2.COLOR_BGR2YUV)

		# equalize the histogram of the Y channel
		img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
		#img_yuv[:,:,1] = cv2.equalizeHist(img_yuv[:,:,1])
		#img_yuv[:,:,2] = cv2.equalizeHist(img_yuv[:,:,2])
	

		# convert the YUV image back to RGB format
		x = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
		
		x = cv2.resize(x, (img_cols, img_rows)).astype(np.float)

	
	mean_pixel = np.array([103.939, 116.779, 123.68])
	for c in range(3):
		x[:, :, c] += - mean_pixel[c]
	

		
	x = x.transpose((2, 0, 1))
	x = np.expand_dims(x, axis=0)
	return x

	
	
	
def filereader_train(fname):

	color_type = 3
	if color_type == 1:
		x = cv2.imread(fname, 0)
	elif color_type == 3:
		x = cv2.imread(fname)
			
	return preprocess( x, True )
	

def filereader_test(fname):

	color_type = 3
	if color_type == 1:
		x = cv2.imread(fname, 0)
	elif color_type == 3:
		x = cv2.imread(fname)
			
	return preprocess( x, False )
		
			
def dataGenerator(y, batch_size, fnames, augment=True):

	#read and preprocess first file to figure out the image dimensions
	sample_file = filereader_test(fnames[0])
	new_img_colours, new_img_rows, new_img_cols = sample_file.shape[1:]	


	
	while 1:
		j = -1
		
		if not y is None:
			o = np.arange(len(fnames))
			random.shuffle(o)
			fnames = fnames[o]
			y = y[o]
			
					
		for i in xrange(int(np.ceil(1.0*len(fnames)/batch_size))):

			j+= 1
			
			this_chunk_size = len(fnames[j*batch_size: (j+1)*batch_size])
			
			
			X = np.zeros((this_chunk_size, new_img_colours, new_img_rows, new_img_cols))
			for q in xrange(this_chunk_size):
				if not augment:
					X[q] = filereader_train(fnames[j*batch_size+q])
				else:
					X[q] = filereader_test(fnames[j*batch_size+q])			

				
			i2 = i - j

			if not y is None:	
				yield X[i2*batch_size:(i2+1)*batch_size], y[i*batch_size:(i+1)*batch_size]

			#test set
			else:
				yield X[i2*batch_size:(i2+1)*batch_size]					

					






def cifar_model(img_rows, img_cols, img_channels=1):
	d = 0.5

	model = Sequential()
	
	init = 'glorot_normal'
	fd = 3 #filter dim

	model.add(Convolution2D(64, fd, fd, border_mode='same', init=init,
							input_shape=(img_channels, img_rows, img_cols)))
	model.add(Activation('relu'))
	#model.add(LeakyReLU())
	model.add(Dropout(d))
	model.add(Convolution2D(64, fd, fd, init=init))
	model.add(Activation('relu'))
	#model.add(LeakyReLU())
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(d))

	model.add(Convolution2D(128, fd, fd, border_mode='same', init=init))
	model.add(Activation('relu'))
	#model.add(LeakyReLU())
	model.add(Dropout(d))
	model.add(Convolution2D(128, fd, fd, init=init))
	model.add(Activation('relu'))
	#model.add(LeakyReLU())
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(d))
	
	model.add(Convolution2D(256, fd, fd, border_mode='same', init=init))
	model.add(Activation('relu'))
	#model.add(LeakyReLU())
	model.add(Dropout(d))
	model.add(Convolution2D(256, fd, fd, init=init))
	model.add(Activation('relu'))
	#model.add(LeakyReLU())
	model.add(MaxPooling2D((5,5), strides=(5,5)))
	model.add(Dropout(d))	

	'''
	model.add(Convolution2D(256, fd, fd, border_mode='same', init=init))
	model.add(Activation('relu'))
	#model.add(LeakyReLU())
	model.add(Dropout(d))
	model.add(Convolution2D(256, fd, fd, init=init))
	model.add(Activation('relu'))
	#model.add(LeakyReLU())
	model.add(MaxPooling2D((4,4), strides=(4,4)))
	model.add(Dropout(d))	
	'''

	model.add(Flatten())
	model.add(Dense(80, init='he_uniform'))
	#model.add(Activation('relu'))
	model.add(LeakyReLU())
	model.add(Dropout(d))
	model.add(Dense(80, init='he_uniform'))
	#model.add(Activation('relu'))	
	model.add(LeakyReLU())
	model.add(Dropout(d))
	model.add(Dense(10, activation='softmax', init='zero'))
	
	# Learning rate is changed to 0.001
	#sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
	#model.compile(optimizer=sgd, loss='categorical_crossentropy')
	
	model.compile(optimizer=Adam(lr=1e-4), loss='categorical_crossentropy')
	
	return model
	
	


def vgg_19_model(img_rows, img_cols, color_type=1):

	d0 = 0.5
	d = 0.5

	model = Sequential()
	model.add(ZeroPadding2D((1,1),input_shape=(color_type, img_rows, img_cols)))
	model.add(Convolution2D(64, 3, 3, activation='relu', trainable=False))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(64, 3, 3, activation='relu', trainable=False ))
	model.add(MaxPooling2D((2,2), strides=(2,2)))

	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(128, 3, 3, activation='relu', trainable=False))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(128, 3, 3, activation='relu', trainable=False))
	model.add(MaxPooling2D((2,2), strides=(2,2)))

	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(256, 3, 3, activation='relu', trainable=False))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(256, 3, 3, activation='relu', trainable=False))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(256, 3, 3, activation='relu', trainable=False))
	#model.add(Dropout(d0))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(256, 3, 3, activation='relu', trainable=False))
	model.add(MaxPooling2D((2,2), strides=(2,2)))
	#model.add(Dropout(d0))

	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, 3, 3, activation='relu', trainable=True))
	model.add(Dropout(d0))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, 3, 3, activation='relu', trainable=True))
	model.add(Dropout(d0))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, 3, 3, activation='relu', trainable=True))
	model.add(Dropout(d0))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, 3, 3, activation='relu', trainable=True))
	model.add(MaxPooling2D((2,2), strides=(2,2)))
	model.add(Dropout(d0))

	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, 3, 3, activation='relu', trainable=True))
	model.add(Dropout(d0))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, 3, 3, activation='relu', trainable=True))
	model.add(Dropout(d0))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, 3, 3, activation='relu'))
	model.add(Dropout(d0))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, 3, 3, activation='relu'))
	model.add(MaxPooling2D((2,2), strides=(2,2)))

	model.add(Dropout(d))
	
	model.add(Flatten())
	model.add(Dense(80, activation='linear', init='he_normal', trainable=True)) #, W_regularizer=l2(l=0.0005) , W_regularizer=l2(l2_reg), b_regularizer=l2(l2_reg)
	model.add(LeakyReLU())
	#model.add(BatchNormalization())
	model.add(Dropout(d))
	model.add(Dense(80, activation='linear',  init='normal', trainable=True)) #, W_regularizer=l2(l=0.0005) , W_regularizer=l2(l2_reg), b_regularizer=l2(l2_reg)
	model.add(LeakyReLU())
	#model.add(BatchNormalization())
	model.add(Dropout(d))	

	model.add(Dense(10, activation='softmax', init='zero'))  #### Instead of 4096 classes,  I have got 10 classes.


	f = h5py.File('vgg19_weights.h5')
	
	print('vgg19')
	print(len(model.layers))
	
	k2 = -1
	for k in xrange(f.attrs['nb_layers']):

		g = f['layer_{}'.format(k)]
		weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
		if k2 > 0 and 'maxpooling2d' in model.layers[k2-1].name and 'dropout' in model.layers[k2].name :
			k2 += 1
		if k2 > 0 and 'relu' in model.layers[k2+1].name:
			k2 += 1
		if k2 > 0 and 'batch' in model.layers[k2+1].name:
			k2 += 0 #1	
			
		if k2 > 0 and 'convolution2d' in model.layers[k2-2].name and 'relu' in model.layers[k2-1].name and 'dropout' in model.layers[k2].name :
			k2 += 1	
		if k2 > 0 and 'convolution2d' in model.layers[k2-1].name and 'dropout' in model.layers[k2].name :
			k2 += 1	
			
		k2 +=1	
		
		
		#print(model.layers[k2].name), k2
		if 'dense' in model.layers[k2].name:
			w = model.layers[k2].get_weights()
			s1 = len(w[0])
			s2 = len(w[1])
			#model.layers[k2].set_weights(zip(weights[0:s1],weights[0:s2]))
			#print(s1, s2)
			print(weights[0][0:s1, 0:s2].shape, weights[1][0:s2].shape)
			if 'dense_1' in model.layers[k2].name:
				w0 = weights[0]
				for t in xrange(4):
					w0 = np.append(w0, np.multiply((1.0+0.03*t), weights[0]), axis=0)
				print(w0.shape)	
				model.layers[k2].set_weights([w0[0:s1, 0:s2], weights[1][0:s2]])
				
				#model.layers[k2].set_weights([np.reshape(weights[0], (weights[0].shape[0]*4, weights[0].shape[1]/4))[0:s1, 0:s2], weights[1][0:s2]])
				#model.layers[k2].set_weights([weights[0][0:s1, 0:s2], weights[1][0:s2]])
				#model.layers[k2].set_weights([weights[0][:, ::4], weights[1][::4]])
				
				
			if 'dense_2' in model.layers[k2].name:
				model.layers[k2].set_weights([weights[0][0:s1, 0:s2], weights[1][0:s2]])
				#model.layers[k2].set_weights([weights[0][::4, ::4], weights[1][::4]])	

			'''
			if 'dense_3' in model.layers[k2].name:
				print('init layer 3')
				model.layers[k2].set_weights([np.multiply(4.0, weights[0][0:s1, 0:s2]), np.multiply(1.0, weights[1][0:s2])])
				#model.layers[k2].set_weights([weights[0][-s1:, -s2:], weights[1][0:s2]])
				#model.layers[k2].set_weights([weights[0][::4, ::4], weights[1][::4]])	
				
			
			if 'dense_4' in model.layers[k2+3].name:
				print('init layer 4')
				w = model.layers[k2+3].get_weights()
				s1 = len(w[0])
				s2 = len(w[1])
				model.layers[k2+3].set_weights([weights[0][0:s1, 0:s2], weights[1][0:s2]])
			'''
				
		else:
			model.layers[k2].set_weights(weights)	
	f.close()
	print('Model loaded.')

	

	
	

	# Learning rate is changed to 0.001
	#sgd = SGD(lr=1e-4, decay=1e-6, momentum=0.9, nesterov=True)
	#model.compile(optimizer=sgd, loss='categorical_crossentropy')
	
	#model.compile(optimizer=Adagrad(lr=1e-4), loss='categorical_crossentropy')
	#model.compile(optimizer=Adam(lr=3e-7), loss='categorical_crossentropy')
	#model.compile(optimizer=Adagrad(lr=5e-5), loss='categorical_crossentropy')
	model.compile(optimizer=Adam(lr=3e-5), loss='categorical_crossentropy')
	
	
	#print(model.summary())
	return model    
    
    
    

def vgg_std16_model(img_rows, img_cols, color_type=1):

	d0 = 0.5
	d1 = 0.5

	model = Sequential()
	model.add(ZeroPadding2D((1, 1), input_shape=(color_type,
												 img_rows, img_cols)))
	model.add(Convolution2D(64, 3, 3, activation='relu', trainable=False))
	#model.add(PReLU())
	#model.add(Dropout(d1))
	model.add(ZeroPadding2D((1, 1)))
	model.add(Convolution2D(64, 3, 3, activation='relu', trainable=False))
	#model.add(PReLU())
	model.add(MaxPooling2D((2, 2), strides=(2, 2)))
	#model.add(Dropout(d1))

	model.add(ZeroPadding2D((1, 1)))
	model.add(Convolution2D(128, 3, 3, activation='relu', trainable=False))
	#model.add(PReLU())
	#model.add(Dropout(d1))
	model.add(ZeroPadding2D((1, 1)))
	model.add(Convolution2D(128, 3, 3, activation='relu', trainable=False))
	#model.add(PReLU())
	model.add(MaxPooling2D((2, 2), strides=(2, 2)))
	#model.add(Dropout(d1))

	model.add(ZeroPadding2D((1, 1)))
	model.add(Convolution2D(256, 3, 3, activation='relu', trainable=False))
	#model.add(PReLU())
	#model.add(Dropout(d0))
	model.add(ZeroPadding2D((1, 1)))
	model.add(Convolution2D(256, 3, 3, activation='relu', trainable=False))
	#model.add(PReLU())
	#model.add(Dropout(d0))
	model.add(ZeroPadding2D((1, 1)))
	model.add(Convolution2D(256, 3, 3, activation='relu', trainable=False))
	#model.add(PReLU())
	model.add(MaxPooling2D((2, 2), strides=(2, 2)))
	#model.add(Dropout(d0))

	model.add(ZeroPadding2D((1, 1)))
	model.add(Convolution2D(512, 3, 3, activation='relu', trainable=True)) 
	#model.add(PReLU())
	#model.add(LeakyReLU())
	model.add(Dropout(d1))	  
	model.add(ZeroPadding2D((1, 1)))
	model.add(Convolution2D(512, 3, 3, activation='relu', trainable=True))
	#model.add(PReLU())
	#model.add(LeakyReLU())
	model.add(Dropout(d1))	  
	model.add(ZeroPadding2D((1, 1)))
	model.add(Convolution2D(512, 3, 3, activation='relu', trainable=True))
	#model.add(PReLU())
	#model.add(LeakyReLU())
	model.add(MaxPooling2D((2, 2), strides=(2, 2)))
	model.add(Dropout(d1))
	
	
	model.add(ZeroPadding2D((1, 1)))
	model.add(Convolution2D(512, 3, 3, activation='relu', trainable=True))
	#model.add(PReLU())
	#model.add(LeakyReLU())	
	model.add(Dropout(d1))
	model.add(ZeroPadding2D((1, 1)))
	model.add(Convolution2D(512, 3, 3, activation='relu', trainable=True))
	#model.add(PReLU())
	#model.add(LeakyReLU())
	model.add(Dropout(d1))
	model.add(ZeroPadding2D((1, 1)))
	model.add(Convolution2D(512, 3, 3, activation='relu', trainable=True))
	#model.add(PReLU())
	#model.add(LeakyReLU())
	model.add(MaxPooling2D((2, 2), strides=(2, 2)))
	
	model.add(Dropout(d1))

	d = 0.5


	
	l2_reg = 0.0001
	model.add(Flatten())
	model.add(Dense(512, activation='linear', init='he_normal', trainable=True)) #, W_regularizer=l2(l=0.0005) , W_regularizer=l2(l2_reg), b_regularizer=l2(l2_reg)
	#model.add(PReLU())
	model.add(LeakyReLU())
	#model.add(BatchNormalization())
	model.add(Dropout(d))
	model.add(Dense(128, activation='linear',  init='he_normal', trainable=True)) #, W_regularizer=l2(l=0.0005) , W_regularizer=l2(l2_reg), b_regularizer=l2(l2_reg)
	#model.add(PReLU())
	model.add(LeakyReLU())
	#model.add(BatchNormalization())
	model.add(Dropout(d))	


	model.add(Dense(10, activation='softmax', init='zero'))  #### Instead of 4096 classes,  I have got 10 classes.


	f = h5py.File('vgg16_weights.h5')
	
	print(len(model.layers))
	
	k2 = -1
	for k in xrange(f.attrs['nb_layers']):

		g = f['layer_{}'.format(k)]
		weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
		if k2 > 0 and 'maxpooling2d' in model.layers[k2-1].name and 'dropout' in model.layers[k2].name :
			k2 += 1
		if k2 > 0 and 'relu' in model.layers[k2+1].name:
			k2 += 1
		if k2 > 0 and 'batch' in model.layers[k2+1].name:
			k2 += 0 #1	
			
		if k2 > 0 and 'convolution2d' in model.layers[k2-2].name and 'relu' in model.layers[k2-1].name and 'dropout' in model.layers[k2].name :
			k2 += 1	
		if k2 > 0 and 'convolution2d' in model.layers[k2-1].name and 'dropout' in model.layers[k2].name :
			k2 += 1	
			
		k2 +=1	
		
		
		#print(model.layers[k2].name), k2
		
		
		if 'dense' in model.layers[k2].name:

			w = model.layers[k2].get_weights()
			s1 = len(w[0])
			s2 = len(w[1])
			#model.layers[k2].set_weights(zip(weights[0:s1],weights[0:s2]))
			#print(s1, s2)
			print(weights[0][0:s1, 0:s2].shape, weights[1][0:s2].shape)
			if 'dense_1' in model.layers[k2].name:
				w0 = weights[0]
				for t in xrange(4):
					w0 = np.append(w0, np.multiply((1.0+0.03*t), weights[0]), axis=0)
				print(w0.shape)	
				model.layers[k2].set_weights([w0[0:s1, 0:s2], weights[1][0:s2]])
				
				#model.layers[k2].set_weights([np.reshape(weights[0], (weights[0].shape[0]*4, weights[0].shape[1]/4))[0:s1, 0:s2], weights[1][0:s2]])
				#model.layers[k2].set_weights([weights[0][0:s1, 0:s2], weights[1][0:s2]])
				#model.layers[k2].set_weights([weights[0][:, ::4], weights[1][::4]])
				
				
			if 'dense_2' in model.layers[k2].name:
				model.layers[k2].set_weights([weights[0][0:s1, 0:s2], weights[1][0:s2]])
				#model.layers[k2].set_weights([weights[0][::4, ::4], weights[1][::4]])	
	

			'''
			if 'dense_3' in model.layers[k2].name:
				print('init layer 3')
				model.layers[k2].set_weights([np.multiply(4.0, weights[0][0:s1, 0:s2]), np.multiply(1.0, weights[1][0:s2])])
				#model.layers[k2].set_weights([weights[0][-s1:, -s2:], weights[1][0:s2]])
				#model.layers[k2].set_weights([weights[0][::4, ::4], weights[1][::4]])	
				
			
			if 'dense_4' in model.layers[k2+3].name:
				print('init layer 4')
				w = model.layers[k2+3].get_weights()
				s1 = len(w[0])
				s2 = len(w[1])
				model.layers[k2+3].set_weights([weights[0][0:s1, 0:s2], weights[1][0:s2]])
			'''
				
		else:
			model.layers[k2].set_weights(weights)	
	f.close()
	
	print(model.summary())
		
	print('Model loaded.')

	

	

	# Learning rate is changed to 0.001
	#sgd = SGD(lr=1e-4, decay=1e-6, momentum=0.9, nesterov=True)
	#model.compile(optimizer=sgd, loss='categorical_crossentropy')
	
	#model.compile(optimizer=Adagrad(lr=1e-4), loss='categorical_crossentropy')
	#model.compile(optimizer=Adam(lr=1e-6), loss='categorical_crossentropy')
	#model.compile(optimizer=Adagrad(lr=5e-5), loss='categorical_crossentropy')
	model.compile(optimizer=Adam(lr=3e-5), loss='categorical_crossentropy')
	
	
	#print(model.summary())
	return model


def get_validation_predictions(train_data, predictions_valid):
	pv = []
	for i in range(len(train_data)):
		pv.append(predictions_valid[i])
	return pv







#y = run_cross_validation(3)
def run_cross_validation(nfolds=1):
	# input image dimensions
	#img_rows, img_cols = 224, 224 #224, 300 #192, 256 #324, 432# 240, 320 #   324, 432 # 150, 200 # 120, 160 # 96, 128 #168, 224 #   225, 300 #150, 200 # 162, 216 #120, 160 #135, 180  #240, 320 #96, 128 # 240, 320 #288, 384#225, 300 #216, 288 #  216, 288 #24, 224 #124, 172 #162, 216 #144, 192 #144, 192 #96, 128 #
	# color type: 1 - grey, 3 - rgb
	color_type_global = 3
	batch_size = 16 #2
	epoch_size = 3200# 6400 #
	nb_epoch = 50
	random_state = 7000 #5 #3 #51
	restore_from_last_checkpoint = 0
	
	print 'batch_size', batch_size

	d = pd.read_csv('driver_imgs_list.csv')
	d['fname'] = d.apply(lambda x : 'train/'+x[1] + '/' + x[2], axis=1)
	d['class'] = d['classname'].astype('category').cat.codes
	
	nb_classes = d['class'].nunique()
	
	unique_drivers = d.subject.unique()
	
	test_df = pd.read_csv('sample_submission.csv')
	fnames_test = test_df['img'].apply(lambda x : 'test/'+x).values
	#yfull_train = dict()
	
	yfull_test = np.zeros((nfolds, len(fnames_test), 10))
	kf = KFold(len(unique_drivers), n_folds=nfolds, shuffle=True, random_state=random_state)
	num_fold = 0
	sum_score = 0
	

	
	for train_drivers, valid_drivers in kf: #list(kf)[dofold:dofold+1]:
		#model = create_model_v1(img_rows, img_cols, color_type_global)
		#model = cifar_model(img_rows, img_cols, color_type_global)
		model = vgg_std16_model(img_rows, img_cols, color_type_global)
		#model = vgg_19_model(img_rows, img_cols, color_type_global)
		

		
		unique_list_train = [unique_drivers[i] for i in train_drivers]
		np.random.shuffle(unique_list_train)
		
		unique_list_valid = [unique_drivers[i] for i in valid_drivers]

		num_fold += 1
		print('Start KFold number {} from {}'.format(num_fold, nfolds))
		print('Train drivers: ', unique_list_train)
		print('Test drivers: ', unique_list_valid)

		kfold_weights_path = os.path.join('cache', 'weights_kfold_' + str(num_fold) + '.h5')
		if not os.path.isfile(kfold_weights_path) or restore_from_last_checkpoint == 0:
  
			
			fnames_es_valid = d[d.subject.isin(unique_list_train[0:3])]['fname'].values
			print('validation', unique_list_train[0:3])
			Y_es_valid = np_utils.to_categorical(d[d.subject.isin(unique_list_train[0:3])]['class'].values, nb_classes) 

	
			callbacks = [
				EarlyStopping(monitor='val_loss', patience=2, verbose=0),
				ModelCheckpoint(kfold_weights_path, monitor='val_loss', save_best_only=True, verbose=0),
			]
		

							  
			fnames_train = d[d.subject.isin(unique_list_train[3:])]['fname'].values
			Y_train = np_utils.to_categorical(d[d.subject.isin(unique_list_train[3:])]['class'].values, nb_classes)		
			
			trainGenerator = dataGenerator(Y_train,batch_size, fnames_train, augment=True)
			validGenerator = dataGenerator(Y_es_valid, batch_size, fnames_es_valid, augment=False)

			model.fit_generator(trainGenerator, samples_per_epoch = epoch_size, nb_epoch = nb_epoch, verbose=1,callbacks=callbacks, validation_data=validGenerator, nb_val_samples=Y_es_valid.shape[0], class_weight=None, max_q_size=10, nb_worker=16 ) # show_accuracy=True, nb_worker=1 
  
			
		if os.path.isfile(kfold_weights_path):
			model.load_weights(kfold_weights_path)
		
		fnames_cv = d[d.subject.isin(unique_list_valid)]['fname'].values
		Y_cv = np_utils.to_categorical(d[d.subject.isin(unique_list_valid)]['class'].values, nb_classes) 
		print(Y_cv.shape)
		cvGenerator = dataGenerator(None, batch_size, fnames_cv, augment=False)
		
		predictions_valid = model.predict_generator(cvGenerator, val_samples = Y_cv.shape[0]) 
		score = log_loss(Y_cv, predictions_valid)
		print('Score log_loss: ', score)
		sum_score += score*Y_cv.shape[0]		
		
		testGenerator = dataGenerator(None, batch_size, fnames_test, augment=False)
		predictions_test = model.predict_generator(testGenerator, val_samples = len(fnames_test)) 
		yfull_test[num_fold-1] = predictions_test
		
	
		with open('fold_'+str(num_fold)+'.csv','wb') as f:
			np.savetxt(f, yfull_test[num_fold-1], delimiter=",", fmt='%1.6f')
	
		ssub = pd.read_csv('sample_submission.csv')
		for i in xrange(0,10):
			ssub['c'+str(i)] = np.mean(yfull_test[0:num_fold, :, i], axis=0)
	
		ssub.to_csv('submission4.csv', index=False)	
		print 'written submission'	
	
	
	return yfull_test
