import glob
import xml.etree.ElementTree as ET
from shutil import copyfile
import cv2
"""

Created on Tue Jun 25 09:48:40 2019

@author: mitchell
"""

base_path = "/home/mitchell/Eyesea-download_data/Proposal/eyesea/people/xuwe421/Delete"
folder = "wells_test_"

put_fodler = "/home/mitchell/YOLO_data/data"
put_name = "wells_test_"

valid_name = "wells_val_"

files = sorted(glob.glob(base_path + '/' + folder + 'y/*'))

train_cfg_name = "/home/mitchell/pytorchYolo/cfg/train_test_locs/train_wells.txt"
valid_cfg_name = "/home/mitchell/pytorchYolo/cfg/train_test_locs/val_wells.txt"

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
            
            image_path = base_path + "/" + folder + "x" + "/" + filename
            print(image_path)
            if not success_count % success_skip == 0:
                image_put = put_fodler + "/" + put_name + "images/" + filename_jpg
                cfg = train_cfg_name
            else:
                image_put = put_fodler + "/" + valid_name + "images/" + filename_jpg
                cfg = valid_cfg_name
            img = cv2.imread(image_path)
            width = img.shape[1]
            height = img.shape[0]
            if not write_img:
                #copyfile(image_path, image_put)
                #cv2.rectangle(img, (int(x_min),int(y_min)),(int(x_max),int(y_max)), (0,0,255))
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
            #print(put_data)
            """
            cv2.circle(img, (int(x_min+box_width/2), int(y_min+box_height/2)), 1, (255,0,0))
            cv2.rectangle(img, (int(x_min),int(y_min)),(int(x_max),int(y_max)), (0,0,255))
            cv2.imshow("img", img)
            cv2.waitKey(1)
            """
            #print(x_min, x_max, width, box_width, (x_min+box_width/2)/width)
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
                
            
    