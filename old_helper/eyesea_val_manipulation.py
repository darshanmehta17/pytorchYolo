#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 13:52:02 2019

@author: mitchell
"""

import glob
import xml.etree.ElementTree as ET
from shutil import copyfile
import cv2
"""

Created on Tue Jun 25 09:48:40 2019

@author: mitchell
"""
width = 1280
height = 960

base_path = "/home/mitchell/Eyesea-download_data/Proposal/eyesea/people/xuwe421/Delete"
folder = "train_"

put_fodler = "/home/mitchell/YOLO_data/data"
put_name = "eyesea_train_"

valid_name = "eyesea_val_"

files = sorted(glob.glob(base_path + '/' + folder + 'y/*'))

train_cfg_name = "/home/mitchell/pytorchYolo/cfg/train_test_locs/train_eyesea_orpc.txt"
valid_cfg_name = "/home/mitchell/pytorchYolo/cfg/train_test_locs/val_eyesea_orpc.txt"

success_skip = 10 # for validation
success_count = 0
for _file in files:
    tree = ET.parse(_file)
    root = tree.getroot()
    count = 0
    for child in root:

        write_img = False
        if child.tag == 'filename':
            filename = child.text
            filename_jpg = filename.replace('png', 'jpg')
            
        if child.tag == 'object':
            count += 1
            for form in child.findall("./bndbox/xmin"):
                x_min = float(form.text)
            for form in child.findall("./bndbox/ymin"):
                y_min = float(form.text)
            for form in child.findall("./bndbox/xmax"):
                x_max = float(form.text)
            for form in child.findall("./bndbox/ymax"):
                y_max = float(form.text) 
            
            image_path = base_path + "/" + put_name + "x" + "/" + filename
            if not success_count % success_skip == 0:
                image_put = put_fodler + "/" + put_name + "images/" + filename_jpg
                cfg = train_cfg_name
            else:
                image_put = put_fodler + "/" + valid_name + "images/" + filename_jpg
                cfg = valid_cfg_name
            
            if not write_img:
                #copyfile(image_path, image_put)
                img = cv2.imread(image_path)
                cv2.rectangle(img, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (255,0,0), 2)
                #cv2.imshow("img", img)
                #cv2.waitKey(1)
                cv2.imwrite(image_put, img)
                f = open(cfg, "a+")
                #print(image_put)
                f.write(image_put + '\n')
                f.close()
                write_img = True
                
                
                
                
            
            """ Generate label """
            if not success_count % success_skip == 0:
                label_path = put_fodler + "/" + put_name + "labels" + "/" + _file.split('/')[-1].strip(".xml") + ".txt"
            else:
                label_path = put_fodler + "/" + valid_name + "labels" + "/" + _file.split('/')[-1].strip(".xml") + ".txt"
            f = open(label_path, "a+")
            #put_data = str(0) + " " + str(x_min/width) + " " + str(y_min/height) + " " + str((x_max-x_min)/width) + " " + str((y_max - y_min)/height)
            box_width = x_max - x_min
            box_height = y_max - y_min
            put_data = str(0) + " " + str((x_min+box_width/2)/width) + " " + str((y_min+box_height/2)/height) + " " + str(box_width/width) + " " + str(box_height/height)
            print(put_data)
            put_data += '\n'
            f.write(put_data)
            f.close()
            success_count

            """
            print(image_path, " ", image_put)
            print()
            print(label_path, " ",  put_data)
            print()
            """
    success_count+=1
                
            
    