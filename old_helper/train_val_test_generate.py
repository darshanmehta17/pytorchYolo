import glob
import argparse
import os

home_directory  = os.path.expanduser('~')
parser = argparse.ArgumentParser("count mine images helper script")
parser.add_argument("base_path", help="path to directory with images and labels", default = home_directory + "/data")
parser.add_argument("--save_path", help="path to directory to save .txt files for YOLO training", default = home_directory + "/data")
parser.add_argument("--test", help="To generate test images or not", default = False, type=bool)
parser.add_argument("--test_frequecy", help="Frequency of test images to take (e.g. 10 = 1/10 of total images are used for testing)", default = 5, type=int)
parser.add_argument("--val_frequecy", help="Frequency of val images to take (e.g. 10 = 1/10 of total images are used for val)", default = 4, type=int)
parser.add_argument("--image_folder_name", help='Image folder name (i.e. train_images). Code expects a similar labels folder to be present', default = "train_images")

args = parser.parse_args()
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