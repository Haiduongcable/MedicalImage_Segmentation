import numpy as np 
import time 
import os 
import random 
import shutil

#Split Kvaiseg 
path_kvaidb = "/home/haiduong/Documents/Project 3/Kvasir-SEG"
path_images = path_kvaidb + "/images"
path_masks = path_kvaidb + "/masks"

l_n_image = os.listdir(path_images)

random.shuffle(l_n_image)

path_train = "data/Kval/train"
path_test = "data/Kval/test"
path_val = "data/Kval/val"

#Train 800
for n_image in l_n_image[:800]:
    path_image_src = path_images + "/" + n_image
    path_image_des = path_train + "/images/" + n_image

    path_mask_src = path_masks + "/" + n_image
    path_mask_des = path_train + "/masks/" + n_image
    shutil.copy(path_image_src, path_image_des)
    shutil.copy(path_mask_src, path_mask_des)

#Val 100
for n_image in l_n_image[800:900]:
    path_image_src = path_images + "/" + n_image
    path_image_des = path_val + "/images/" + n_image

    path_mask_src = path_masks + "/" + n_image
    path_mask_des = path_val + "/masks/" + n_image
    shutil.copy(path_image_src, path_image_des)
    shutil.copy(path_mask_src, path_mask_des)

#Test 100
for n_image in l_n_image[900:]:
    path_image_src = path_images + "/" + n_image
    path_image_des = path_test + "/images/" + n_image

    path_mask_src = path_masks + "/" + n_image
    path_mask_des = path_test + "/masks/" + n_image
    shutil.copy(path_image_src, path_image_des)
    shutil.copy(path_mask_src, path_mask_des)

