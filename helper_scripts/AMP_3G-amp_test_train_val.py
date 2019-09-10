#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 09:00:21 2019

@author: mitchell
"""
from shutil import copyfile

all_imgs_file = open("/home/mitchell/YOLO_ws/pytorchYolo/cfg/train_test_locs/full_amp_3G-amp_set.txt", 'r')
all_files = []

test_txt_file = open("/home/mitchell/YOLO_ws/pytorchYolo/cfg/train_test_locs/AMP_3G-AMP_updated_training/amp_3G-amp_test.txt", 'a+')
train_txt_file = open("/home/mitchell/YOLO_ws/pytorchYolo/cfg/train_test_locs/AMP_3G-AMP_updated_training/amp_3G-amp_train.txt", 'a+')
val_txt_file = open("/home/mitchell/YOLO_ws/pytorchYolo/cfg/train_test_locs/AMP_3G-AMP_updated_training/amp_3G-amp_val.txt", 'a+')

img_save_path = "/home/mitchell/YOLO_data/data"

test_images_folder = "amp_3G-amp_train_test_images"
train_val_images_folder = "amp_3G-amp_train_test_images"


#Take 20% of images as test images
test_amount = 5

#Take 20% of NON-TEST images as val images
val_amount = 5
val_count = 0

for i, img_file in enumerate(all_imgs_file):
    if img_file in all_files:
        continue
    all_files.append(img_file)
    if i % test_amount == 0:
        #Take test anount
        img_name = img_file.split('/')[-1]
        copyfile(img_file.rstrip(), img_save_path + '/' + test_images_folder + '/' + img_name.rstrip())
        label_name = img_file.replace('images', 'labels').replace('.jpg', '.txt')
        copyfile(label_name.rstrip(), img_save_path + '/' + test_images_folder.replace('images', 'labels') + '/' + img_name.rstrip().replace('.jpg', '.txt'))
        test_txt_file.write(img_save_path + '/' + test_images_folder + '/' + img_name.rstrip() + '\n')
        
    else:
        img_name = img_file.split('/')[-1]
        copyfile(img_file.rstrip(), img_save_path + '/' + train_val_images_folder + '/' + img_name.rstrip())
        label_name = img_file.replace('images', 'labels').replace('.jpg', '.txt')
        copyfile(label_name.rstrip(), img_save_path + '/' + train_val_images_folder.replace('images', 'labels') + '/' + img_name.rstrip().replace('.jpg', '.txt'))        
        val_count += 1
        if val_count % val_amount == 0:
            val_txt_file.write(img_save_path + '/' + train_val_images_folder + '/' + img_name.rstrip() + '\n')
        else:
            train_txt_file.write(img_save_path + '/' + train_val_images_folder + '/' + img_name.rstrip() + '\n')
    
        