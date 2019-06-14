from time import time
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2
from utils import *
import argparse
from constants import *
import os
from darknet import Darknet
import pickle as pkl
import pandas as pd
import random


def arg_parse():
    """
    Parse command line arguments passed to the detector module.

    """
    parser = argparse.ArgumentParser(description='YOLO v3 Detection Module')

    parser.add_argument("--images", help="Image / Directory containing images to perform detection upon",
                        default="data/test_images", type=str)
    parser.add_argument("--output", help="Image / Directory to store detections to", default="data/output", type=str)
    parser.add_argument("--batch_size", help="Batch size", default=1, type=int)
    parser.add_argument("--conf_thresh", help="Object Confidence to filter predictions", default=0.5, type=float)
    parser.add_argument("--nms_thresh", help="IOU threshold for non-max suppression", default=0.4, type=float)
    parser.add_argument("--cfg", help=" Path to the config file", default="cfg/yolov3-amp.cfg", type=str)
    parser.add_argument("--weights", help="Path to the weights file", default="weights/yolov3.weights", type=str)
    parser.add_argument("--img_size", help="Input resolution of the network. Increase to increase accuracy. "
                                           "Decrease to increase speed", default=416, type=int)
    parser.add_argument("--data_cfg", help="Path to the data cfg file", default="cfg/amp.data", type=str)

    return parser.parse_args()


if __name__ == '__main__':
    args = arg_parse()  # parse the command line arguments
    print(args)

    # Get the arguments passed via command line
    images = args.images
    batch_size = args.batch_size
    conf_thresh = args.conf_thresh
    nms_thresh = args.nms_thresh
    data_cfg_file = args.data_cfg
    cfg_file = args.cfg
    weights_file = args.weights
    output_dir = args.output
    img_size = args.img_size

    start = 0
    gpu = torch.cuda.is_available()  # check and see if GPU processing is available

    data_config = parse_data_cfg(data_cfg_file)  # parse the data config file
    num_classes = int(data_config[CONF_CLASSES])
    classes = load_classes(data_config[CONF_NAMES])  # load the list of classes

    # Initialize the network and load the weights from the file
    print('Loading the network...')
    model = Darknet(cfg_file)
    model.load_weights(weights_file)
    model.set_input_dimension(img_size)

    if gpu:
        model.cuda()

    print('Network loaded successfully.')

    # Set the model in evaluation mode
    model.eval()

    # Read the image list from the directory
    start_time_read_dir = time()
    try:
        imlist = [os.path.join(os.path.realpath('.'), images, image) for image in os.listdir(images)]
    except NotADirectoryError:  # Thrown when the user supplies path to a single image instead of a directory
        imlist = []
        imlist.append(os.path.join(os.path.realpath('.'), images))
    except FileNotFoundError:
        print("No file or directory found with the name {}".format(images))
        exit()
    end_time_read_dir = time()

    # Create the directory to save the output if it doesn't already exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load the images from the paths
    start_time_load_batch = time()
    loaded_images = [cv2.imread(impath) for impath in imlist]

    # Move images to PyTorch variables
    im_batches = list(map(prepare_image, loaded_images, [(img_size, img_size) for x in range(len(imlist))]))

    # Save the list of dimensions or original images for remapping the bounding boxes later
    im_dim_list = [(image.shape[1], image.shape[0]) for image in loaded_images]
    im_dim_list = torch.FloatTensor(im_dim_list).repeat(1, 2)

    if gpu:
        im_dim_list = im_dim_list.cuda()

    # Create the batches
    leftover = 0
    if len(im_dim_list) % batch_size:
        leftover = 1

    if batch_size != 1:
        num_batches = len(imlist) // batch_size + leftover
        im_batches = [torch.cat((im_batches[i * batch_size: min((i + 1) * batch_size, len(im_batches))]))
                      for i in range(num_batches)]
    end_time_load_batch = time()

    # Perform the detections
    write = 0
    start_time_det_loop = time()
    for idx, batch in enumerate(im_batches):
        start_time_batch = time()
        if gpu:
            batch = batch.cuda()  # move the batch to the GPU
        with torch.no_grad():
            prediction = model(Variable(batch), gpu)
        prediction = filter_transform_predictions(prediction, num_classes, conf_thresh, nms_thresh)
        end_time_batch = time()

        if type(prediction) == int:
            for im_num, image in enumerate(imlist[idx * batch_size: min((idx + 1) * batch_size, len(imlist))]):
                im_id = idx * batch_size + im_num
                print("{0:20s} predicted in {1:6.3f} seconds".format(image.split("/")[-1],
                                                                     (end_time_batch - start_time_batch) / batch_size))
                print("{0:20s} {1:s}".format("Objects Detected:", ""))
                print("----------------------------------------------------------")
            continue

        prediction[:, 0] += idx * batch_size  # transform the image reference index from batch level to imlist level

        if not write:
            output = prediction
            write = 1
        else:
            output = torch.cat((output, prediction))


        for im_num, image in enumerate(imlist[idx * batch_size: min((idx + 1) * batch_size, len(imlist))]):
            im_id = idx * batch_size + im_num
            objs = [classes[int(x[-1])] for x in output if int(x[0]) == im_id]
            print("{0:20s} predicted in {1:6.3f} seconds".format(image.split("/")[-1],
                                                                 (end_time_batch - start_time_batch) / batch_size))
            print("{0:20s} {1:s}".format("Objects Detected:", " ".join(objs)))
            print("----------------------------------------------------------")

        if gpu:
            torch.cuda.synchronize()

    try:
        output
    except NameError:
        exit()

    # Translate the dimensions of the bounding box from the input size of the network to the original size of the image
    im_dim_list = torch.index_select(im_dim_list, 0, output[:, 0].long())
    scaling_factor = torch.min(img_size / im_dim_list, 1)[0].view(-1, 1)

    output[:, [1, 3]] -= (img_size - scaling_factor * im_dim_list[:, 0].view(-1, 1)) / 2
    output[:, [2, 4]] -= (img_size - scaling_factor * im_dim_list[:, 1].view(-1, 1)) / 2

    # Also undo the scaling introduced by the image_to_letterbox function
    output[:, 1:5] /= scaling_factor

    # Clip the bounding boxes which have boundaries outside the image
    for i in range(output.shape[0]):
        output[i, [1, 3]] = torch.clamp(output[i, [1, 3]], 0.0, im_dim_list[i, 0])
        output[i, [2, 4]] = torch.clamp(output[i, [2, 4]], 0.0, im_dim_list[i, 1])

    end_time_det_loop = time()

    # Draw the boxes and save the images
    start_time_draw_box = time()
    colors = pkl.load(open("pallete.pickle", "rb"))
    class_color_pair = [(class_name, random.choice(colors)) for class_name in classes]  # assign a color to each class

    def write(x, results):
        """
        Draws the bounding box in x over the image in results with a random color.
        :param x: Bounding box to be drawn.
        :param results: Images over which the bounding box needs to be drawn
        :return img: The final image containing the bounding boxes drawn.
        """
        c1 = tuple(x[1:3].int())  # top-left coordinates
        c2 = tuple(x[3:5].int())  # bottom-right coordinates
        img = results[int(x[0])]  # get the image corresponding to the bounding box
        cls = int(x[-1])  # get the class index
        label, color = "{0}".format(class_color_pair[cls][0]), class_color_pair[cls][1]
        cv2.rectangle(img, c1, c2, color, 1)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
        c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
        cv2.rectangle(img, c1, c2, color, -1)
        cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225, 255, 255], 1)
        return img


    list(map(lambda x: write(x, loaded_images), output))  # draw the boxes on the images
    # Generate output file names
    detection_names = pd.Series(imlist).apply(lambda x: "{}/det_{}".format(output_dir, x.split("/")[-1]))
    list(map(cv2.imwrite, detection_names, loaded_images))  # write the files
    end_time_draw_box = time()

    print("SUMMARY")
    print("----------------------------------------------------------")
    print("{:25s}: {}".format("Task", "Time Taken (in seconds)"))
    print()
    print("{:25s}: {:2.3f}".format("Reading addresses", end_time_read_dir - start_time_read_dir))
    print("{:25s}: {:2.3f}".format("Loading batch", end_time_load_batch - start_time_load_batch))
    print("{:25s}: {:2.3f}".format("Detection (" + str(len(imlist)) + " images)",
                                   end_time_det_loop - start_time_det_loop))
    print("{:25s}: {:2.3f}".format("Drawing Boxes", end_time_draw_box - start_time_draw_box))
    print("{:25s}: {:2.3f}".format("Average time_per_img", (end_time_draw_box - start_time_load_batch) / len(imlist)))
    print("----------------------------------------------------------")

    torch.cuda.empty_cache()
