#!/usr/bin/env python2.7
import cv2
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorchYolo.constants import *
from torch.autograd import Variable
from pytorchYolo.utils import parse_cfg, create_network, transform_predictions, filter_transform_predictions


class Darknet(nn.Module):
    """
    # TODO: Fill documentation.
    """
    def __init__(self, cfgfile):
        """
        Initializes the Darknet network according to the configuration file provided.
        :param cfgfile: path to the network configuration file
        """
        super(Darknet, self).__init__()
        self._blocks = parse_cfg(cfgfile)  # read the file and get the different components
        self._net_info, self._module_list = create_network(self._blocks)  # create a network from the above components
        # TODO: Acquire height, width, batch_size and perform necessary tasks


    def set_input_dimension(self, height):
        """
        Set the 'height' value of the network info which would be later used by the YOLO layer.
        :param height: Specifies the height value
        """
        assert height > 32, "Height must be greater than 32."
        self._net_info[LPROP_HEIGHT] = height


    def forward(self, data_tensor, gpu=False):
        """
        Implements the forward pass of the Darknet network on the input data_tensor
        and returns the detections made across all scales.
        :param data_tensor: input data on which the forward pass must performed
        :param gpu: Boolean flag to denote if compute must be on CPU (False) or GPU(True). Default is False.
        :return detections: List of bounding boxes as predicted by the Darknet.
        """
        # TODO: Move more compute to GPU if possible
        # Acquire a list of network modules to work with.
        # We skip the first element since it contains network
        # information and isn't part of the forward pass.
        modules = self._blocks[1:]
        outputs = {}  # We cache the outputs for the route layer
        write = False
        detections = []

        if gpu:
            data_tensor = data_tensor.cuda()

        for index, module in enumerate(modules):
            layer_type = module[LPROP_TYPE]  # get the type of the layer

            if layer_type in [LAYER_CONVOLUTIONAL, LAYER_UPSAMPLE]:
                # Run the input through the corresponding layer at that index in the
                # list of network objects acquired from create_network function.
                # NOTE: This will pass the image through the entire Sequential Layer.
                data_tensor = self._module_list[index](data_tensor)

            elif layer_type == LAYER_ROUTE:
                # Here we target two cases:
                #
                # 1) When we route the outputs of only one layer.
                # 2) When we route the outputs of multiple layers.
                #
                # In the latter case, we would have to merge the outputs
                # from all the layers along the channel (depth) dimension for which
                # we would use the torch.cat function with the dim=1 because
                # the tensors have a format batch_size * channels * height * width.
                # TODO: Optimize this section of the code
                layers = list(map(int, module[LPROP_LAYERS].split(',')))

                if layers[0] > 0:
                    layers[0] = layers[0] - index  # switch to negative indexing

                if len(layers) == 1:
                    data_tensor = outputs[index + layers[0]]  # fetch the output of the necessary layer
                else:
                    if layers[1] > 0:
                        layers[1] = layers[1] - index  # switch to negative indexing

                    map1 = outputs[index + layers[0]]
                    map2 = outputs[index + layers[1]]

                    data_tensor = torch.cat((map1, map2), dim=1)

            elif layer_type == LAYER_SHORTCUT:
                from_layer_idx = int(module[LPROP_FROM])
                # Since the activation is always linear so we hard code that.
                data_tensor = outputs[index - 1] + outputs[index + from_layer_idx]

            elif layer_type == LAYER_YOLO:
                # Get the anchors from the inserted DetectionLayer in the module list
                anchors = self._module_list[index][0].anchors  # 0 refers to the first object in the Sequential Layer

                # Get the input dimensions
                input_dim = int(self._net_info[LPROP_HEIGHT])  # assumes a square image

                # Get the number of classes
                num_classes = int(module[LPROP_CLASSES])

                # Transform the predictions
                data_tensor = transform_predictions(predictions=data_tensor.data, input_dim=input_dim,
                                                    anchors=anchors, num_classes=num_classes, gpu=gpu)

                if not write:
                    detections = data_tensor
                    write = True
                else:
                    detections = torch.cat((detections, data_tensor), 1)

            else:
                assert False, "Unknown layer encountered."

            outputs[index] = data_tensor

        return detections

    def load_weights(self, weights_file):
        """
        A utility function to load the weights from a file into the model.

        The official weights file (*.weights) is a binary file which contains weights stored in a
        serial fashion only for the convolution and the batch norm layers. The values are stored
        as floats and have no metadata linking them to the layers. So we use the sizes of the weights
        and the biases of each layer in the module list we generated using create_network to read
        the data from the file.

        The first five values contain header information such major version, minor version, number
        of images seen, etc. The rest are the weights. If a convolution layer has a batch norm layer
        following it, then the weights are present as follows:

        <batch_norm biases> <batch_norm weights> <batch_norm running_mean> <batch_norm running_variance> <conv weights>

        If not, then the weights are present as follows:
        <conv biases> <conv weights>

        :param weights_file: Path to the file containing the model weights.
        """
        # Open the weights file
        fp = open(weights_file, "rb")

        # The first 5 values contain header information
        # 1. Major version number
        # 2. Minor Version Number
        # 3. Subversion number
        # 4,5. Images seen by the network (during training)
        header = np.fromfile(fp, dtype=np.int32, count=5)
        self.header = torch.from_numpy(header)
        self.seen = self.header[3]

        # The rest of the values contain the weights in Float32 format
        weights = np.fromfile(fp, dtype=np.float32)

        ptr = 0  # to keep a track of how much has been read from the file
        for index, block in enumerate(self._blocks[1:]):
            layer_type = block[LPROP_TYPE]

            # Since the weights are stored only for the convolution layer, we only process those layers.
            if layer_type == LAYER_CONVOLUTIONAL:
                layer = self._module_list[index]
                convolution_layer = layer[0]  # first element in the Sequential Layer in the Convolution Layer

                # We check and see if batch normalization layer exists. If it does, we load the weights and
                # biases of the batch norm layer, and if not, we load the biases of the convolution layer
                try:
                    batch_normalize = int(block[LPROP_BATCH_NORMALIZE])
                except:
                    batch_normalize = 0

                if batch_normalize:
                    bn_layer = layer[1]  # get the batch norm layer if it exists
                    num_bn_biases = bn_layer.bias.numel()  # get the number of biases of the batch norm layer

                    # Load the weights
                    bn_biases = torch.from_numpy(weights[ptr: (ptr + num_bn_biases)])
                    ptr += num_bn_biases

                    bn_weights = torch.from_numpy(weights[ptr: (ptr + num_bn_biases)])
                    ptr += num_bn_biases

                    bn_running_mean = torch.from_numpy(weights[ptr: (ptr + num_bn_biases)])
                    ptr += num_bn_biases

                    bn_running_var = torch.from_numpy(weights[ptr: (ptr + num_bn_biases)])
                    ptr += num_bn_biases

                    # Reshape the weights as required by the model implementation
                    bn_biases = bn_biases.view_as(bn_layer.bias.data)
                    bn_weights = bn_weights.view_as(bn_layer.weight.data)
                    bn_running_mean = bn_running_mean.view_as(bn_layer.running_mean)
                    bn_running_var = bn_running_var.view_as(bn_layer.running_var)

                    # Copy the data into the model
                    bn_layer.bias.data.copy_(bn_biases)
                    bn_layer.weight.data.copy_(bn_weights)
                    bn_layer.running_mean.copy_(bn_running_mean)
                    bn_layer.running_var.copy_(bn_running_var)
                else:
                    num_conv_biases = convolution_layer.bias.numel()  # get the number of biases of the conv layer

                    # Load the weights
                    conv_biases = torch.from_numpy(weights[ptr: (ptr + num_conv_biases)])
                    ptr = ptr + num_conv_biases

                    # Reshape the loaded weights according to the dimensions of the model weights
                    conv_biases = conv_biases.view_as(convolution_layer.bias.data)

                    # Copy the data into the model
                    convolution_layer.bias.data.copy_(conv_biases)

                # Load the weights of the convolution layer
                num_conv_weights = convolution_layer.weight.numel()  # get the number of weights
                conv_weights = torch.from_numpy(weights[ptr: (ptr + num_conv_weights)])
                ptr += num_conv_weights

                # Reshape the weights as required and copy it into the model
                conv_weights = conv_weights.view_as(convolution_layer.weight.data)
                convolution_layer.weight.data.copy_(conv_weights)

        fp.close()  # close the weights file


def get_test_input(image_path, dim):
    """
    Reads image from file and prepares for input to the Darknet.
    :param image_path: Path to the input image file.
    :param dim: Dimension to which the image must be resized.
    :return img_: Image in the form of torch variable.
    """
    img = cv2.imread(image_path)
    img = cv2.resize(img, (dim, dim))  # Resize to the input dimension
    img_ = img[:, :, ::-1].transpose((2, 0, 1))  # BGR -> RGB | H * W * C -> C * H * W
    img_ = img_[np.newaxis, :, :, :] / 255.0  # Add a channel at 0 (for batch) | Normalise
    img_ = torch.from_numpy(img_).float()  # Convert to float
    img_ = Variable(img_)  # Convert to Variable
    return img_
