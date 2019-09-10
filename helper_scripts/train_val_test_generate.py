import glob
import argparse
import os
from helper_argparse import ArgLoader

arl = ArgLoader()
args = arl.args
base_path = args.base_path
save_path = args.save_path
test = args.test
test_frequency = args.test_frequecy
val_frequency = args.val_frequecy
image_folder_name = args.image_folder_name
label_folder_name = image_folder_name.replace('images', 'labels')


train_file = open(save_path + "/train.txt", "a+")
val_file = open(save_path + "/val.txt", "a+")
if test:
    test_file = open(save_path + "/test.txt", "a+")
    
images = sorted(glob.glob(base_path + "/" + image_folder_name + "/*.jpg"))

val_count = 0
val_num = val_frequency

test_count = 0
test_num = test_frequency



for image in images:
    
    text_file = base_path + "/" + label_folder_name + "/" + image.split('/')[-1].replace('.jpg', '.txt')
    label = open(text_file, 'r')
    label_len = len(label.readlines())
    if label_len > 0:
        if test_count % test_num == 0:
            test_file.write(image + '\n')
        else:
            if val_count % val_num == 0: 
                val_file.write(image + '\n')
            else:
                train_file.write(image + '\n')
                
                
            val_count+=1
        test_count+=1