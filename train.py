import os
import re
import csv
import json
from PIL import Image
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from glob import glob
from PIL import Image
np.random.seed(123)
from sklearn.preprocessing import label_binarize
from sklearn.metrics import confusion_matrix
import itertools
import warnings
import keras
from keras.utils.np_utils import to_categorical # used for converting labels to one-hot-encoding
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout,Input,Average,Conv2DTranspose,SeparableConv2D,dot,UpSampling2D,Add, Flatten,Concatenate,Multiply,Conv2D, MaxPooling2D,Activation,AveragePooling2D, ZeroPadding2D,GlobalAveragePooling2D,multiply,DepthwiseConv2D,ZeroPadding2D,GlobalAveragePooling2D
from tensorflow.keras import backend as K
from tensorflow.keras.layers import concatenate ,Lambda
import itertools
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import LearningRateScheduler, ModelCheckpoint 
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import ResNet50,VGG19,VGG16,DenseNet121,DenseNet169,InceptionResNetV2
from tensorflow.keras.losses import BinaryCrossentropy,CategoricalCrossentropy
import numpy as np
from skimage.morphology import square,binary_erosion,binary_dilation,binary_opening,binary_closing
from skimage.morphology import erosion, dilation, opening, closing, white_tophat
from skimage.morphology import black_tophat, skeletonize, convex_hull_image
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras import regularizers
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
from math import sqrt, ceil
from PIL import Image
import numpy as np
from tqdm import tqdm_notebook as tqdm
import tensorflow as tf
import cv2
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tqdm import tqdm
from glob import glob
import tifffile as tif
from sklearn.model_selection import train_test_split
import os
import numpy as np
import cv2
from model import msrf
from model import *
import tensorflow as tf
from tensorflow.keras.callbacks import *
from tensorflow.keras.optimizers import Adam, Nadam
from tensorflow.keras.metrics import *
from glob import glob
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import skimage.io
from skimage.transform import rescale, resize, downscale_local_mean
from skimage.color import rgb2gray
import imutils
from model.msrf import msrf
from model.loss_metric import *
import argparse
from tqdm import tqdm
from numpy_dataloader import DataLoader
parser = argparse.ArgumentParser()
parser.add_argument('--batchsize', type=int, default = 4 , help='initial batch size')
parser.add_argument('--finetune', default = True, help='initial batch size')
opt = parser.parse_args()

gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=1.0)
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session()

from glob import glob

def get_optimizer():
 
    adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    return adam

def train(epochs, batch_size):
    
    
    max_val_dice= -1
    G = msrf(input_size=(256,256,3), input_size_2=(256,256,1))
    # if opt.finetune:
    #     G.load_weights("./lass_weight.h5")
    # G.summary()
    optimizer = get_optimizer()
    G.compile(optimizer = optimizer, loss = {'x':seg_loss,'edge_out':'binary_crossentropy','pred4':seg_loss,'pred2':seg_loss},loss_weights={'x':2.,'edge_out':1.,'pred4':1. , 'pred2':1.})
    for e in range(1, epochs+1):
        print ('-'*15, 'Epoch %d' % e, '-'*15,batch_size)
        #sp startpoint
        dataset = DataLoader(batch_size = batch_size, size_image = 256)
        
        num_batch = dataset.get_num_batch()
        X_val, X_edge_val, Y_val, Y_edge_val = dataset.get_val_dataset()
        X_test, X_edge_test, Y_test, Y_edge_test = dataset.get_test_dataset()
        for sp in tqdm(range(0,num_batch,1)):
            X_batch, X_edge_batch, Y_batch, Y_edge_batch = dataset.get_batch_train(index = sp)
            G.train_on_batch([X_batch,X_edge_batch],[Y_batch,Y_edge_batch,Y_batch,Y_batch])

        y_pred,_,_,_ = G.predict([X_val,X_edge_val],batch_size=5)
        y_pred = (y_pred >=0.5).astype(int)
        res = mean_dice_coef(Y_val,y_pred)
        print("Mean Dice Score Val: ", res)
        #Log on testing 
        y_pred_test,_,_,_ = G.predict([X_test,X_edge_test],batch_size=5)
        y_pred_test = (y_pred_test >=0.5).astype(int)
        res_test = mean_dice_coef(Y_test,y_pred_test)
        print("Mean Dice Score Test: ", res_test)
        
       
        G.save('/kaggle/working/Log_weight/lass_weight.h5')
        if(res > max_val_dice):
            max_val_dice = res
            G.save('/kaggle/working/Log_weight/best_val_dice.h5')
            print('New Val_Dice HighScore',res)            
            

train(125,opt.batchsize)