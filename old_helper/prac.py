#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 15:52:00 2019

@author: mitchell
"""
import glob

f = open("/home/mitchell/pytorchYolo/cfg/train_test_locs/all_amp_3G-amp_with_negatives_train.txt", "r")
write = open("/home/mitchell/pytorchYolo/cfg/train_test_locs/all_amp_3G-amp_with_negatives_train_NEW.txt", "a+")
labels = glob.glob("/home/mitchell/AMP_YOLO_data/train_data/3G-AMP_labels/*.txt")

print(labels)

for line in f:
    if line in labels:
        print(line)
        write.write(line)