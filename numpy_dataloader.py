import cv2 
import numpy as np 
import os 
import time 
import imgaug as ia
import imgaug.augmenters as iaa
import cv2 
import random





class DataLoader:
    def __init__(self, batch_size, size_image, path_dataset):
        self.path_train = path_dataset + "/train"
        self.path_val = path_dataset + "/val"
        self.path_test = path_dataset + "/test"
        self.augmentation_mask = iaa.Sequential([
            iaa.Fliplr(0.5),
            iaa.Crop(percent=(0, 0.1)),
            iaa.Flipud(0.5),
            iaa.Affine(rotate=90)
        ], random_order=True)
        self.size_image = size_image
        self.batch_size = batch_size
        self.l_train_images, self.l_train_masks = self.load_data_resize(self.path_train)
        print(len(self.l_train_images), len(self.l_train_masks))
        self.l_aug_images, self.l_aug_masks = self.augmentation_mask(images = self.l_train_images,\
                                                                    segmentation_maps = self.l_train_masks)
        self.l_val_images, self.l_val_masks = self.load_data_resize(self.path_val)
        self.l_test_images, self.l_test_masks = self.load_data_resize(self.path_test)
        
        
    def process_data(self,image, mask):
        edge_image = cv2.Canny(image,10,1000)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        edge_mask = cv2.Canny(mask,10,1000)

        image = np.asarray(image).astype(np.float32)/255.0
        mask = np.asarray(mask).astype(np.float32)/255.0

        edge_image = np.asarray(edge_image).astype(np.float32)/255.0
        edge_mask = np.asarray(edge_mask).astype(np.float32)/255.0

        mask=(mask >=0.5).astype(int)
        return image, edge_image, mask, edge_mask

    def load_data_resize(self, path_data):
        images = []
        masks = []
        for n_image in os.listdir(path_data + "/images"):
            path_image = path_data + "/images/" + n_image
            path_mask = path_data + "/masks/" + n_image
            image = cv2.imread(path_image)
            image = cv2.resize(image, (self.size_image, self.size_image))
            
            mask = cv2.imread(path_mask)
            mask = cv2.resize(mask, (self.size_image, self.size_image))
            images.append(image)
            masks.append(mask)
        return images, masks

    def get_num_batch(self):
        return len(self.l_train_images) // self.batch_size


    def get_batch_train(self,index):
        X_batch = []
        Y_batch = []
        X_edge_batch = []
        Y_edge_batch = []
        batch_start = index * self.batch_size
        if (index + 1) * self.batch_size > len(self.l_train_images):
            batch_end = len(self.l_train_images)
        else:
            batch_end = (index + 1) * self.batch_size
        X_batch_list = self.l_aug_images[batch_start: batch_end]
        Y_batch_list = self.l_aug_masks[batch_start: batch_end]
        for index in range(len(X_batch_list)):
            image, edge_image, mask, edge_mask =\
             self.process_data(X_batch_list[index], Y_batch_list[index])
            X_batch.append(image)
            Y_batch.append(mask)
            X_edge_batch.append(edge_image)
            Y_edge_batch.append(edge_mask)
        X_batch = np.array(X_batch).astype(np.float32)
        X_edge_batch = np.array(X_edge_batch).astype(np.float32)
        X_edge_batch = np.expand_dims(X_edge_batch,axis=3)

        Y_batch = np.array(Y_batch).astype(np.float32)
        Y_batch = np.expand_dims(Y_batch,axis=3)
        
        Y_edge_batch = np.array(Y_edge_batch).astype(np.float32)
        Y_edge_batch = np.expand_dims(Y_edge_batch,axis=3)
        return X_batch, X_edge_batch, Y_batch, Y_edge_batch


    def get_val_dataset(self):
        X_batch = []
        Y_batch = []
        X_edge_batch = []
        Y_edge_batch = []
        X_batch_list = self.l_val_images
        Y_batch_list = self.l_val_masks
        for index in range(len(X_batch_list)):
            image, edge_image, mask, edge_mask =\
             self.process_data(X_batch_list[index], Y_batch_list[index])
            X_batch.append(image)
            Y_batch.append(mask)
            X_edge_batch.append(edge_image)
            Y_edge_batch.append(edge_mask)
        X_batch = np.array(X_batch).astype(np.float32)
        X_edge_batch = np.array(X_edge_batch).astype(np.float32)

        Y_batch = np.array(Y_batch).astype(np.float32)
        Y_batch = np.expand_dims(Y_batch,axis=3)
        
        Y_edge_batch = np.array(Y_edge_batch).astype(np.float32)
        Y_edge_batch = np.expand_dims(Y_edge_batch,axis=3)


        return X_batch, X_edge_batch, Y_batch, Y_edge_batch

    def get_test_dataset(self):
        X_batch = []
        Y_batch = []
        X_edge_batch = []
        Y_edge_batch = []
        X_batch_list = self.l_test_images
        Y_batch_list = self.l_test_masks
        for index in range(len(X_batch_list)):
            image, edge_image, mask, edge_mask =\
             self.process_data(X_batch_list[index], Y_batch_list[index])
            X_batch.append(image)
            Y_batch.append(mask)
            X_edge_batch.append(edge_image)
            Y_edge_batch.append(edge_mask)
        X_batch = np.array(X_batch).astype(np.float32)
        X_edge_batch = np.array(X_edge_batch).astype(np.float32)

        Y_batch = np.array(Y_batch).astype(np.float32)
        Y_batch = np.expand_dims(Y_batch,axis=3)
        
        Y_edge_batch = np.array(Y_edge_batch).astype(np.float32)
        Y_edge_batch = np.expand_dims(Y_edge_batch,axis=3)


        return X_batch, X_edge_batch, Y_batch, Y_edge_batch

if __name__ == '__main__':
    dataset = DataLoader(batch_size=4, size_image=256)
    print(dataset.get_num_batch())
    X_batch, X_edge_batch, Y_batch, Y_edge_batch = dataset.get_batch_train(index = 0)
    print(np.shape(X_batch), np.shape(X_edge_batch), np.shape(Y_batch), np.shape(Y_edge_batch))
