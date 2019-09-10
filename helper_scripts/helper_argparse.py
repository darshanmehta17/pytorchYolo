#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 13:48:58 2019

@author: mitchell
"""

#!/usr/bin/env python3
import argparse
import os

class ArgLoader:
    @property
    def args(self):
        """
        Parse command line arguments passed to the detector module.
    
        """
        
        home_directory  = os.path.expanduser('~')
        parser = argparse.ArgumentParser("generate train test and val .txt files")
        parser.add_argument("base_path", help="path to directory with images and labels", default = home_directory + "/data")
        parser.add_argument("--save_path", help="path to directory to save .txt files for YOLO training", default = home_directory + "/data")
        parser.add_argument("--test", help="To generate test images or not", default = False, type=bool)
        parser.add_argument("--test_frequecy", help="Frequency of test images to take (e.g. 10 = 1/10 of total images are used for testing)", default = 5, type=int)
        parser.add_argument("--val_frequecy", help="Frequency of val images to take (e.g. 10 = 1/10 of total images are used for val)", default = 4, type=int)
        parser.add_argument("--image_folder_name", help='Image folder name (i.e. train_images). Code expects a similar labels folder to be present', default = "train_images")
 
    
        return parser.parse_args()      