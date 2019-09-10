#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 08:45:59 2019

@author: mitchell
"""
import json
import glob
import cv2
import os

base_path = "/home/mitchell/AMP_YOLO_data/train_data/test_images/"
save_path = "/home/mitchell/YOLO_data/data/"
folder_name = "test_"

train_cfg_name = "/home/mitchell/pytorchYolo/cfg/train_test_locs/WAMP_test.txt"
valid_cfg_name = "/home/mitchell/pytorchYolo/cfg/train_test_locs/WAMP_test.txt"

img_files = sorted(glob.glob(base_path + '*.jpg'))
print(base_path + '*.jpg')
#count = 0 
success_skip = 6 # for validation
success_count = 0

val_files = open("all_amp_3G-amp_with_negatives_train.txt", 'r').readlines()
for i, line in enumerate(val_files):
    val_files[i] = line.rstrip().split('/')[-1]
    
for img_name in img_files:
    _file = img_name.replace(".jpg", ".json")
    img = cv2.imread(img_name)
    #print(img_name)
    
    img_name = img_name.strip('.jpg') + '.jpg'
    _file = _file.strip('.json') + '.json'
    #print(img_name)    
    img_width = img.shape[1]
    img_height= img.shape[0]
    
    if os.path.exists(_file):
        with open(_file) as json_file:
            data = json.load(json_file)
            if len(data["shapes"]) > 0:
                for i in range(len(data["shapes"])):
                    label = data["shapes"][i]["label"]
                    points = data["shapes"][i]["points"]
                    left = (int(points[0][0]), int(points[0][1]))
                    right = (int(points[1][0]), int(points[1][1]))
                    width = float(abs((right[0] - left[0])))
                    height = float(abs((right[1] - left[1])))
                    x_vals = (left[0], right[0])
                    y_vals = (left[1], right[1])
                    center_x = float(min(x_vals)+ width/2)
                    center_y = float(min(y_vals) + height/2)
                    center = (center_x, center_y)
                    write_line = str(0) + " " + str(center_x/img_width) + " " + str(center_y/img_height) + " " + str(width/img_width) + " " + str(height/img_height) + "\n"
                    label_path = save_path + folder_name + "labels/" + _file.split('/')[-1].replace(".json", ".txt")
                    #print(write_line)
                    f = open(label_path, "a+")
                    f.write(write_line)
                    f.close()
                    image_path = save_path + folder_name + "images/" + img_name.split('/')[-1]
                
                    if success_count % success_skip != 0:
                        
                        f = open(train_cfg_name, "a+")
                        f.write(image_path  + '\n') 
                        f.close()
                    else:
                        
                        f = open(valid_cfg_name, "a+")
                        f.write(image_path  + '\n' )        
                        f.close()
                #print(image_path)
                cv2.imwrite(image_path, img)
                #cv2.circle(img, center, 10, (255, 255, 255))
                #cv2.circle(img, right, 10, (255, 0, 0))
                #cv2.imshow("img", img)
                #cv2.waitKey(0)
                
            else:
                label_path = save_path + folder_name + "labels/" + _file.split('/')[-1].replace(".json", ".txt")
                print(2, label_path)
                #print(write_line)
                f = open(label_path, "a+")
                f.close()
                image_path = image_path.strip('.jpg') + '.jpg'
                if success_count % success_skip != 0:
                    
                    f = open(train_cfg_name, "a+")
                    f.write(image_path  + '\n') 
                    f.close()
                else:
                    
                    f = open(valid_cfg_name, "a+")
                    f.write(image_path  + '\n' )        
                    f.close()
                cv2.imwrite(image_path, img)
    else:
            label_path = save_path + folder_name + "labels/" + _file.split('/')[-1].replace(".json", ".txt")
            #print(3, label_path)
            #print(write_line)
            f = open(label_path, "a+")
            f.close()
            image_path = save_path + folder_name + "images/" + img_name.split('/')[-1]
            if success_count % success_skip != 0:
                
                f = open(train_cfg_name, "a+")
                f.write(image_path  + '\n') 
                f.close()
            else:
                
                f = open(valid_cfg_name, "a+")
                f.write(image_path  + '\n' )        
                f.close()
            cv2.imwrite(image_path, img)    
            
    
    success_count += 1
        
            
            
