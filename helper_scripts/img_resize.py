#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 08:26:04 2019

@author: mitchell
"""

import cv2
import glob
"""

Created on Tue Jun 25 09:48:40 2019

@author: mitchell
"""
width = 352
height = 240

train_name_path = "/home/mitchell/pytorchYolo/cfg/train_test_locs"
train_name = "train_amp_rename_subset"
val_name = "val_amp_rename_subset"

put_fodler = "/home/mitchell/pytorchYolo/cfg/train_test_locs"
put_train_name = "train_amp_rename_subset_cropped"
put_val_name = "val_amp_rename_subset_cropped"


'''
Train data first 
'''
