# Introduction
A custom implementation of the Darknet platform tailored for the 3G-AMP research project under the Applied Physics Laboratory at the Univeristy of Washington. This project can be used to perform inferencing on a pretrained YOLOv3 network.

## Requirements
- CUDA Version: 10.1
- CUDA compilation tools, release 10.1, V10.1.105
- Tensorflow-GPU: 1.13.1
- Torch: 1.0.1.post2
- PyTorch: 1.0.1 (py3.6_cuda10.0.130_cudnn7.4.2_2)
- Cuda Toolkit: 10.0.130
- OpenCV-Python: 3.4.4.19

For a detailed list of all dependencies, please refer to the [cvenv.yml](cvenv.yml) present in the root directory of this repository.

## Generating dataset
To generate an annotated dataset, use the [labelImg](https://github.com/tzutalin/labelImg) tool. Instructions on using the tool can be found at the linked repository page. The labels must be generated using the YOLO configuration provided by the tool to be compatible with the training process. Each ```<image>.jpg``` would now have a corresponding ```<image>.txt``` file associated with it which would contain the list of bounding boxes. An additional file ```classes.txt``` would be generated which would contain the names of the different classes of objects present in the dataset which have been labelled.

Once these files are ready, we move on to create our train, test and validation splits. For each of the splits, we would create a txt file where each line of the file would containing a path to an image file which belongs to the subset. 

**Note**: For this generation process, we usually collect all the images into a single folder and use that folder with the labelImg tool. Check the ```data/images_subset``` and the ```data/labels_subset``` (Present on the APL machine) for an example.

## Training the model
For the training of the model, we could use either the [Darknet](https://github.com/pjreddie/darknet) or any similar implementations such as the  [YOLOv3 by Ultralytics](https://github.com/ultralytics/yolov3). For the model and weights present on this repository, we have used the original Darknet implementation. 

#### Instructions for training with Darknet
1. The network architecture and the training configuration must be specified in the [cfg/yolov3-amp.cfg](cfg/yolov3-amp.cfg) file. This file would then be placed in the ```cfg``` folder under the Darknet project.
2. Next, we create a file which would specify the path to the ```train.txt``` and the ```val.txt``` file which we prepared in the previous section. This file would also specify the number of classes and path to the ```classes.txt``` which we created in the previous section. A demo of this file can be seen in [cfg/amp.data](cfg/amp.data).
3. Download pretrained convolution weights and place them in the root directory of the Darknet project. 
```sh
wget https://pjreddie.com/media/files/darknet53.conv.74
```
4. Train the model using the command:
```sh
./darknet detector train cfg/amp.data cfg/yolov3-amp.cfg darknet53.conv.74
```

# Detection using a pre-trained model

