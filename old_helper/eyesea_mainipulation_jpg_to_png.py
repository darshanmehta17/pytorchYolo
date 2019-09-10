#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 16:56:58 2019

@author: mitchell
"""

import glob
import xml.etree.ElementTree as ET
from shutil import copyfile
"""

Created on Tue Jun 25 09:48:40 2019

@author: mitchell
"""
width = 352
height = 240

base_path = "/home/mitchell/Eyesea-download_data/Proposal/eyesea/people/xuwe421/Delete"
folder = "wells_test_"

put_fodler = "/home/mitchell/YOLO_data/data"
put_name = "wells_test_"

valid_name = "wells_val_"

files = sorted(glob.glob(base_path + '/' + folder + 'y/*'))

train_cfg_name = "/home/mitchell/pytorchYolo/cfg/train_test_locs/train_eyesea_wells_jpg.txt"
valid_cfg_name = "/home/mitchell/pytorchYolo/cfg/train_test_locs/val_eyesea_wells_jpg.txt"

success_skip = 10 # for validation
success_count = 0
for _file in files:
    tree = ET.parse(_file)
    root = tree.getroot()
    for child in root:
        write_img = False
        if child.tag == 'filename':
            filename = child.text
            filename_jpg = filename.replace('.png', '.jpg')
            
        if child.tag == 'object':
            for form in root.findall("./object/bndbox/xmin"):
                x_min = float(form.text)
            for form in root.findall("./object/bndbox/ymin"):
                y_min = float(form.text)
            for form in root.findall("./object/bndbox/xmax"):
                x_max = float(form.text)
            for form in root.findall("./object/bndbox/ymax"):
                y_max = float(form.text) 
                
            
            image_path = base_path + "/" + put_name + "x" + "/" + filename
            if not success_count % success_skip == 0:
                image_put = put_fodler + "/" + put_name + "images/" + filename
                cfg = train_cfg_name
            else:
                image_put = put_fodler + "/" + valid_name + "images/" + filename
                cfg = valid_cfg_name
            
            if not write_img:
                copyfile(image_path, image_put)
                f = open(cfg, "a+")
                f.write(image_put + '\n')
                f.close()
                write_img = True
                
                
                
                
            
            """ Generate label """
            if not success_count % success_skip == 0:
                label_path = put_fodler + "/" + put_name + "labels" + "/" + _file.split('/')[-1].strip(".xml") + ".txt"
            else:
                label_path = put_fodler + "/" + valid_name + "labels" + "/" + _file.split('/')[-1].strip(".xml") + ".txt"
            f = open(label_path, "a+")
            put_data = str(0) + " " + str(x_min/width) + " " + str(y_max/height) + " " + str((x_max-x_min)/width) + " " + str((y_max - y_min)/height)
            f.write(put_data)
            f.close()
            
            """
            print(image_path, " ", image_put)
            print()
            print(label_path, " ",  put_data)
            print()
            """
    success_count+=1
                
            
    