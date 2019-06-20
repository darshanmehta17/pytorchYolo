from time import time, sleep
import torch
from torch.autograd import Variable
import cv2
import utils
import constants 
import os
from darknet import Darknet
import pickle as pkl
import pandas as pd
import random

import glob

class Detector():
    def __init__(self, args):
        #Load and parse args
        
        self.base_path = args.base_path
        
        # Get the arguments passed via command line
        self.images = args.images
        self.videofile = args.video
        self.batch_size = args.batch_size
        self.conf_thresh = args.conf_thresh
        self.nms_thresh = args.nms_thresh
        self.data_cfg_file = args.data_cfg
        #self.cfg_file = args.cfg
        #self.weights_file = args.weights
        self.output_dir = args.output
        self.img_size = args.img_size
        
        data_config = utils.parse_data_cfg(self.data_cfg_file)  # parse the data config file
        self.classes = utils.load_classes(data_config[constants.CONF_NAMES].rstrip())  # load the list of classes
        self.num_classes = len(self.classes)
        self.cfg_file = data_config["cfg"].rstrip()
        self.weights_file = data_config["weights"].rstrip()
        
                
        #Use GPU, if possible
        self.gpu = torch.cuda.is_available()  # check and see if GPU processing is available = 
        
        self.create_model()
        
        self.imlist_len = 0
        
        #Initalize time stats
        self.start_time_det_loop = time()
        self.start_time_read_dir = time()
        self.end_time_read_dir = time()
        self.start_time_load_batch = time()
        self.end_time_load_batch = time()
        self.end_time_det_loop = time()
        self.start_time_draw_box = time()
        self.end_time_draw_box = time()
        
        self.write_pred = False

    
    def print_end_stats(self):
        print("SUMMARY")
        print("----------------------------------------------------------")
        print("{:25s}: {}".format("Task", "Time Taken (in seconds)"))
        print()
        print("{:25s}: {:2.3f}".format("Loading batch", self.end_time_load_batch - self.start_time_load_batch))
        print("{:25s}: {:2.3f}".format("Detection (" + str(self.imlist_len) + " images)",
                                       self.end_time_det_loop - self.start_time_det_loop))
        print("{:25s}: {:2.3f}".format("Drawing Boxes", self.end_time_draw_box - self.start_time_draw_box))
        print("{:25s}: {:2.3f}".format("Average time_per_img", (self.end_time_draw_box - self.start_time_load_batch) / self.imlist_len))
        print("----------------------------------------------------------")
            
    def write(self, x, results):
        """
        Draws the bounding box in x over the image in results with a random color.
        :param x: Bounding box to be drawn.
        :param results: Images over which the bounding box needs to be drawn
        :return img: The final image containing the bounding boxes drawn.
        """
        colors = pkl.load(open("pallete.pickle", "rb"))
        class_color_pair = [(class_name, random.choice(colors)) for class_name in self.classes]  # assign a color to each class
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
    
    def create_model(self):
        # Initialize the network and load the weights from the file
        print('Loading the network...')
        model = Darknet(self.cfg_file)
        model.load_weights(self.weights_file)
        model.set_input_dimension(self.img_size)
    
        if self.gpu:
            model.cuda()
    
        print('Network loaded successfully.')
    
        # Set the model in evaluation mode
        model.eval()
        
        self.model = model
    
    
    def get_model(self):
        return self.model
    
    @property
    def inp_dim(self):
        inp_dim = int(self.img_size)
        assert inp_dim % 32 == 0 
        assert inp_dim > 32
        
        return inp_dim
        
    


class YoloImgRun(Detector):
    
    def load_images(self):
        # Read the image list from the directory
        self.start_time_read_dir = time()
        try:
            imlist = [os.path.join(os.path.realpath('.'), self.images, image) for image in os.listdir(self.images)]
        except NotADirectoryError:  # Thrown when the user supplies path to a single image instead of a directory
            imlist = []
            imlist.append(os.path.join(os.path.realpath('.'), self.images))
        except FileNotFoundError:
            print("No file or directory found with the name {}".format(self.images))
            exit()
        self.end_time_read_dir = time()
    
        # Create the directory to save the output if it doesn't already exist
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
    
        # Load the images from the paths
        self.start_time_load_batch = time()
        loaded_images = [cv2.imread(impath) for impath in imlist]
    
        # Move images to PyTorch variables
        im_batches = list(map(utils.prepare_image, loaded_images, [(self.img_size, self.img_size) for x in range(len(imlist))]))
    
        # Save the list of dimensions or original images for remapping the bounding boxes later
        im_dim_list = [(image.shape[1], image.shape[0]) for image in loaded_images]
        im_dim_list = torch.FloatTensor(im_dim_list).repeat(1, 2)
    
        if self.gpu:
            im_dim_list = im_dim_list.cuda()  
            # Create the batches
        leftover = 0
        if len(im_dim_list) % self.batch_size:
            leftover = 1
    
        if self.batch_size != 1:
            num_batches = len(imlist) // self.batch_size + leftover
            im_batches = [torch.cat((im_batches[i * self.batch_size: min((i + 1) * self.batch_size, len(im_batches))]))
                          for i in range(num_batches)]
        self.end_time_load_batch = time()
            
        self.imlist_len = len(imlist)
        
        return imlist, im_dim_list, loaded_images, im_batches
    
    
    def run(self):         
        #Load the images and image batches
        imlist, im_dim_list, loaded_images, im_batches = self.load_images()
        
        # Perform the detections
        write = 0
        self.start_time_det_loop = time()
        for idx, batch in enumerate(im_batches):
            start_time_batch = time()
            if self.gpu:
                batch = batch.cuda()  # move the batch to the GPU
            with torch.no_grad():
                prediction = self.model(Variable(batch), self.gpu)
            prediction = utils.filter_transform_predictions(prediction, self.num_classes, self.conf_thresh, self.nms_thresh)
            end_time_batch = time()
    
            if type(prediction) == int:
                for im_num, image in enumerate(imlist[idx * self.batch_size: min((idx + 1) * self.batch_size, len(imlist))]):
                    im_id = idx * self.batch_size + im_num
                    print("{0:20s} predicted in {1:6.3f} seconds".format(image.split("/")[-1],
                                                                         (end_time_batch - start_time_batch) / self.batch_size))
                    print("{0:20s} {1:s}".format("Objects Detected:", ""))
                    print("----------------------------------------------------------")
                continue
    
            prediction[:, 0] += idx * self.batch_size  # transform the image reference index from batch level to imlist level
    
            if not write:
                output = prediction
                
                write = 1
            else:
                output = torch.cat((output, prediction))
                
    
    
            for im_num, image in enumerate(imlist[idx * self.batch_size: min((idx + 1) * self.batch_size, len(imlist))]):
                im_id = idx * self.batch_size + im_num
                objs = [self.classes[int(x[-1])] for x in output if int(x[0]) == im_id]
                print("{0:20s} predicted in {1:6.3f} seconds".format(image.split("/")[-1],
                                                                     (end_time_batch - start_time_batch) / self.batch_size))
                print("{0:20s} {1:s}".format("Objects Detected:", " ".join(objs)))
                print("----------------------------------------------------------")
    
            if self.gpu:
                torch.cuda.synchronize()
    
        try:
            output
        except NameError:
            exit()
            
            
            

        # Translate the dimensions of the bounding box from the input size of the network to the original size of the image
        im_dim_list = torch.index_select(im_dim_list, 0, output[:, 0].long())
        scaling_factor = torch.min(self.img_size / im_dim_list, 1)[0].view(-1, 1)
    
        output[:, [1, 3]] -= (self.img_size - scaling_factor * im_dim_list[:, 0].view(-1, 1)) / 2
        output[:, [2, 4]] -= (self.img_size - scaling_factor * im_dim_list[:, 1].view(-1, 1)) / 2
    
        # Also undo the scaling introduced by the image_to_letterbox function
        output[:, 1:5] /= scaling_factor
    
        # Clip the bounding boxes which have boundaries outside the image
        for i in range(output.shape[0]):
            output[i, [1, 3]] = torch.clamp(output[i, [1, 3]], 0.0, im_dim_list[i, 0])
            output[i, [2, 4]] = torch.clamp(output[i, [2, 4]], 0.0, im_dim_list[i, 1])
            
        self.end_time_det_loop = time()
    
        # Draw the boxes and save the images
        self.start_time_draw_box = time()
        
        list(map(lambda x: self.write(x, loaded_images), output))  # draw the boxes on the images
        # Generate output file names
        detection_names = pd.Series(imlist).apply(lambda x: "{}/{}".format(self.output_dir, x.split("/")[-1]))
        list(map(cv2.imwrite, detection_names, loaded_images))  # write the files
        self.end_time_draw_box = time()
        
        self.print_end_stats()
        
        
        torch.cuda.empty_cache()
        
class YoloLiveVideoStream(Detector):        
    def stream_img(self, img):
        
        orig_im = img
        
        img = utils.prepare_image(img, (self.img_size,self.img_size))
        #cv2.rectangle(orig_im,(384,0),(510,128),(0,255,0),3)
        im_dim = torch.FloatTensor((orig_im.shape[1], orig_im.shape[0])).repeat(1,2)
        
        if self.gpu:
            im_dim = im_dim.cuda()
            img = img.cuda()
            
        with torch.no_grad():   
            prediction = self.model(Variable(img), self.gpu)
            prediction = utils.filter_transform_predictions(prediction, self.num_classes, self.conf_thresh, self.nms_thresh)
    
        if type(prediction) == int:
            cv2.imshow("frame", orig_im)
            return
        """
        if not self.write_pred:
            self.output = prediction
            self.write_pred = True
        else:
            self.output = torch.cat((self.output, prediction))
        """
            
        self.output = prediction
        im_dim = im_dim.repeat(self.output.size(0), 1)
        scaling_factor = torch.min(self.img_size/im_dim,1)[0].view(-1,1)
        
        self.output[:,[1,3]] -= (self.img_size - scaling_factor*im_dim[:,0].view(-1,1))/2
        self.output[:,[2,4]] -= (self.img_size - scaling_factor*im_dim[:,1].view(-1,1))/2
        
        self.output[:,1:5] /= scaling_factor
        """
        for i in range(self.output.shape[0]):
            self.output[i, [1,3]] = torch.clamp(self.output[i, [1,3]], 0.0, im_dim[i,0])
            self.output[i, [2,4]] = torch.clamp(self.output[i, [2,4]], 0.0, im_dim[i,1])
        
        """
        list(map(lambda x: self.write(x, orig_im), self.output))
        #print(self.output)
        
        #cv2.rectangle(img,(384,0),(510,128),(0,255,0),3)
        cv2.imshow("frame", orig_im)
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            return
        
        
    def write(self, x, img):
        #print(type(img))
        c1 = tuple(x[1:3].int())
        c2 = tuple(x[3:5].int())
        cls = int(x[-1])
        label = "{0}".format(self.classes[cls])
        colors = pkl.load(open("pallete.pickle", "rb"))
        color = random.choice(colors)
        cv2.rectangle(img, c1, c2,color, 1)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1 , 1)[0]
        c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
        cv2.rectangle(img, c1, c2,color, -1)
        cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225,255,255], 1);
        return img
        
class YoloVideoRun(YoloLiveVideoStream):
    def run(self):
        self.model
        cap = cv2.VideoCapture(self.videofile)
        
        assert cap.isOpened(), 'Cannot capture source'
        
        frames = 0
        start = time()    
        while cap.isOpened():   
            ret, frame = cap.read()
            if ret:    
                
                self.stream_img(frame)
                print("FPS of the video is {:5.2f}".format( frames / (time() - start)))
         
            else:
                break
            
class YoloImageStream(YoloLiveVideoStream):
    def run(self, pause = 0.1):
        full_images = glob.glob(self.images + "/*")
        for frame in full_images: 
            img = cv2.imread(frame)
            self.stream_img(img)   
            sleep(pause)
    

    
    

    