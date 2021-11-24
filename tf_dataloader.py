import tensorflow as tf 
import numpy as np 
import time 
import os 

class Augment(tf.keras.layers.Layer):
    def __init__(self, seed=42):
        super().__init__()

        self.augment_inputs = tf.keras.layers.RandomFlip(mode="horizontal", seed=seed)
        self.aug_1 = tf.keras.layers.RandomRotation(0.2,seed = seed)
        self.aug_2 = tf.keras.layers.RandomFlip(mode="vertical", seed=seed)


        self.augment_labels = tf.keras.layers.RandomFlip(mode="horizontal", seed=seed)
        self.aug_1_label = tf.keras.layers.RandomRotation(0.2, seed = seed)
        self.aug_2_label = tf.keras.layers.RandomFlip(mode="vertical", seed=seed)
        
        self.augment_edgeImage = tf.keras.layers.RandomFlip(mode="horizontal", seed=seed)
        self.aug_1_edgeImage = tf.keras.layers.RandomRotation(0.2, seed = seed)
        self.aug_2_edgeImage = tf.keras.layers.RandomFlip(mode="vertical", seed=seed)
        
        self.augment_edgeMask = tf.keras.layers.RandomFlip(mode="horizontal", seed=seed)
        self.aug_1_edgeMask = tf.keras.layers.RandomRotation(0.2, seed = seed)
        self.aug_2_edgeMask = tf.keras.layers.RandomFlip(mode="vertical", seed=seed)

    def call(self, inputs, labels, edgeImage, edgeMask):
        inputs = self.augment_inputs(inputs)
        inputs = self.aug_1(inputs)
        inputs = self.aug_2(inputs)

        labels = self.augment_labels(labels)
        labels = self.aug_1_label(labels)
        labels = self.aug_2_label(labels)
        
        edgeImage = self.augment_edgeImage(edgeImage)
        edgeImage = self.aug_1_edgeImage(edgeImage)
        edgeImage = self.aug_2_edgeImage(edgeImage)
        
        edgeMask = self.augment_edgeMask(edgeMask)
        edgeMask = self.aug_1_edgeMask(edgeMask)
        edgeMask = self.aug_2_edgeMask(edgeMask)
        
        return inputs, labels, edgeImage, edgeMask


def load_image(path_image, path_mask,\
               path_edgeimage, path_edgemask):
    '''
    resize 256
    '''
    image = tf.io.read_file(path_image)
    image = tf.image.decode_jpeg(image)
    
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, [256, 256])
    image = image / 255.0

    mask = tf.io.read_file(path_mask)
    mask = tf.image.decode_jpeg(mask)
    mask = tf.image.rgb_to_grayscale(mask)
    mask = tf.image.convert_image_dtype(mask, tf.float32)
    mask = tf.image.resize(mask, [256, 256])
    mask = mask / 255.0
    
    edgeImage = tf.io.read_file(path_edgeimage)
    edgeImage = tf.image.decode_jpeg(edgeImage)
    edgeImage = tf.image.convert_image_dtype(edgeImage, tf.float32)
    edgeImage = tf.image.resize(edgeImage, [256, 256])
    edgeImage = edgeImage / 255.0
    
    edgeMask = tf.io.read_file(path_edgemask)
    edgeMask = tf.image.decode_jpeg(edgeMask)
    edgeMask = tf.image.convert_image_dtype(edgeMask, tf.float32)
    edgeMask = tf.image.resize(edgeMask, [256, 256])
    edgeMask = edgeMask / 255.0
    
    return image, mask, edgeImage, edgeMask

def get_loader_train(train_dir, batchsize):
    BUFFER_SIZE = 1000
    path_images = train_dir + "/" + "images"
    path_masks = train_dir + "/" + "masks"
    path_edgeimages = train_dir + "/" + "edge_image"
    path_edgemasks = train_dir + "/" + "edge_mask"
    
    l_name_image = os.listdir(path_images)
    l_path_image = [path_images + "/" + name_image for name_image in l_name_image]
    l_path_mask = [path_masks + "/" + name_image for name_image in l_name_image]
    l_path_edgeimage = [path_edgeimages + "/" + name_image for name_image in l_name_image]
    l_path_edgemask = [path_edgemasks + "/" + name_image for name_image in l_name_image]
    
    
    dataset = tf.data.Dataset.from_tensor_slices((l_path_image, l_path_mask,\
                                                 l_path_edgeimage, l_path_edgemask))
    dataset = dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    train_batches = (dataset.cache().shuffle(BUFFER_SIZE).\
                    batch(batchsize).map(Augment()).\
                    prefetch(buffer_size = BUFFER_SIZE))
    return train_batches


def get_loader_test(test_dir, batchsize):
    BUFFER_SIZE = 1000
    path_images = test_dir + "/" + "images"
    path_masks = test_dir + "/" + "masks"
    path_edgeimages = test_dir + "/" + "edge_image"
    path_edgemasks = test_dir + "/" + "edge_mask"
    
    l_name_image = os.listdir(path_images)
    l_path_image = [path_images + "/" + name_image for name_image in l_name_image]
    l_path_mask = [path_masks + "/" + name_image for name_image in l_name_image]
    l_path_edgeimage = [path_edgeimages + "/" + name_image for name_image in l_name_image]
    l_path_edgemask = [path_edgemasks + "/" + name_image for name_image in l_name_image]
    
    
    dataset = tf.data.Dataset.from_tensor_slices((l_path_image, l_path_mask,\
                                                 l_path_edgeimage, l_path_edgemask))
    dataset = dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    test_batches = (dataset.cache().shuffle(BUFFER_SIZE).\
                    batch(batchsize).\
                    prefetch(buffer_size = BUFFER_SIZE))
    return test_batches, dataset
    