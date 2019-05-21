from __future__ import division

import numpy as np
import torch
import torch.nn as nn

from constants import *
from layers import EmptyLayer, DetectionLayer


def parse_cfg(cfgfile):
    """
    Takes a configuration file

    Returns a list of blocks. Each blocks describes a block in the neural
    network to be built. Block is represented as a dictionary in the list

    Input:
    ------
    cfgfile (str): Path to the configuration file.

    Output:
    -------
    blocks (list): List of blocks in the neural network.

    """
    file = open(cfgfile, 'r')
    lines = file.read().split('\n')  # store the lines in a list
    lines = [x for x in lines if len(x) > 0]  # get read of the empty lines
    lines = [x for x in lines if x[0] != '#']  # get rid of comments
    lines = [x.rstrip().lstrip() for x in lines]  # get rid of fringe whitespaces

    block = {}
    blocks = []

    for line in lines:
        if line[0] == "[":  # This marks the start of a new block
            if len(block) != 0:  # If block is not empty, implies it is storing values of previous block.
                blocks.append(block)  # add it the blocks list
                block = {}  # re-init the block
            block[LPROP_TYPE] = line[1:-1].rstrip()
        else:
            key, value = line.split("=")
            block[key.rstrip()] = value.lstrip()
    blocks.append(block)

    return blocks

def create_network(blocks):
    """
    Takes a list of blocks as returned by the parse_cfg function and
    creates a network using it.

    :param blocks: list of blocks of the neural network
    :return net_info: dictionary consisting of network and training parameters
    :return module_list: nn.ModuleList object consisting of the layers of the network
    """
    net_info = blocks[0]  # Capture the block containing network parameters

    # check if the net_info captured is correct
    assert net_info[LPROP_TYPE] == LAYER_NET, "Could not find the locate the network info layer. " \
                                              "Please check the cfg file."

    module_list = nn.ModuleList()
    prev_filters = 3  # For the 3 channels of an RGB image
    output_filters = []  # to keep track of the output dimensions of each layer
    filters = None

    # Begin the processing of the remaining blocks of the network.
    # Check the type of the module and create a pytorch implementation of it.
    # Append the layer to the module_list
    for index, block in enumerate(blocks[1:]):
        module = nn.Sequential()  # for grouping layers such as conv, activation and batch norm

        if block[LPROP_TYPE] == LAYER_CONVOLUTIONAL:
            activation_type = block[LPROP_ACTIVATION]
            filters = int(block[LPROP_FILTERS])
            padding = int(block[LPROP_PAD])
            kernel_size = int(block[LPROP_SIZE])
            stride = int(block[LPROP_STRIDE])

            if padding:
                pad = (kernel_size - 1) // 2
            else:
                pad = 0

            try:
                batch_normalize = int(block[LPROP_BATCH_NORMALIZE])
                bias = False
            except:
                batch_normalize = 0
                bias = True

            # Add the convolutional layer to module
            conv = nn.Conv2d(prev_filters, filters, kernel_size, stride, pad, bias=bias)
            module.add_module("conv_{}".format(index), conv)

            # Add the Batch Norm Layer to module if needed
            if batch_normalize:
                bn = nn.BatchNorm2d(filters)
                module.add_module("batch_norm_{}".format(index), bn)

            # Check the activation.
            # It is either RELU or a Leaky ReLU for YOLO
            if activation_type == LAYER_LEAKY:
                activn = nn.LeakyReLU(0.1, inplace=True)
                module.add_module("leaky_{0}".format(index), activn)
            elif activation_type == LAYER_RELU:
                activn = nn.ReLU(inplace=True)
                module.add_module("relu_{0}".format(index), activn)

        elif block[LPROP_TYPE] == LAYER_UPSAMPLE:
            stride = int(block[LPROP_STRIDE])
            upsample = nn.Upsample(scale_factor=2, mode="bilinear")  # TODO: Fix the constant scale value
            module.add_module("upsample_{}".format(index), upsample)

        elif block[LPROP_TYPE] == LAYER_ROUTE:
            layers = block[LPROP_LAYERS].split(',')

            # Start  of a route
            start = int(layers[0])

            # end, if there exists one.
            try:
                end = int(layers[1])
            except:
                end = 0

            # Positive notation
            if start > 0:
                start = start - index
            if end > 0:
                end = end - index

            route = EmptyLayer()
            module.add_module("route_{0}".format(index), route)

            if end < 0:
                filters = output_filters[index + start] + output_filters[index + end]
            else:
                filters = output_filters[index + start]

        elif block[LPROP_TYPE] == LAYER_SHORTCUT:
            # shortcut corresponds to skip connection
            shortcut = EmptyLayer()
            module.add_module("shortcut_{}".format(index), shortcut)

        elif block[LPROP_TYPE] == LAYER_YOLO:
            mask = block[LPROP_MASK].split(",")
            mask = [int(x) for x in mask]

            anchors = block[LPROP_ANCHORS].split(",")
            anchors = [int(a) for a in anchors]
            anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]
            anchors = [anchors[i] for i in mask]  # pick the anchors listed in the mask

            detection = DetectionLayer(anchors)
            module.add_module("Detection_{}".format(index), detection)

        else:
            assert False, "Unknown layer encountered"

        module_list.append(module)
        prev_filters = filters  # TODO: Improve the tracking of filters value
        output_filters.append(filters)
    return net_info, module_list


def transform_predictions(predictions, input_dim, anchors, num_classes, gpu=False):
    """
    Since the outputs produced by the YOLO layer are tedious to transform, we first flatten
    the predictions into a 2D array where each 1D array corresponds to a bounding box.

    Next, we offset the center coordinates by the grid offsets and log transform the
    dimensions of the bounding box by the size of the anchor. We also apply sigmoid
    to the confidence score and the class scores.

    :param predictions: The raw predictions from the YOLO layer.
    :param input_dim: The dimension of the input image. Assumes that the image is square.
    :param anchors: A list of anchors used by the detection layer
    :param num_classes: Number of output classes to be predicted
    :param gpu: Boolean flag to denote if compute must be on CPU (False) or GPU(True). Default is False.
    :return predictions: 2D array consisting of transformed bounding box predictions.
    """
    # TODO: Optimize this function, understand the anchors part and improve documentation.
    batch_size = predictions.size(0)
    stride = input_dim // predictions.size(2)
    grid_size = input_dim // stride
    bbox_attrs = 5 + num_classes  # (x, y, w, h, conf.)
    num_anchors = len(anchors)

    # The anchors are expressed as a factor of the stride, so we get the original dimensions
    anchors = [(anchor[0] / stride, anchor[1] / stride) for anchor in anchors]

    predictions = predictions.view(batch_size, bbox_attrs * num_anchors, grid_size * grid_size)
    predictions = predictions.transpose(1, 2).contiguous()  # the dimensions 1 & 2 are swapped
    predictions = predictions.view(batch_size, grid_size * grid_size * num_anchors, bbox_attrs)

    # Now apply sigmoid to the center coordinates, confidence score and the class scores
    predictions[:, :, 0] = torch.sigmoid(predictions[:, :, 0])  # x value
    predictions[:, :, 1] = torch.sigmoid(predictions[:, :, 1])  # y value
    predictions[:, :, 4] = torch.sigmoid(predictions[:, :, 4])  # confidence score
    predictions[:, :, 5:(5 + num_classes)] = torch.sigmoid(predictions[:, :, 5:(5 + num_classes)])  # class scores

    # Create and add the grid offsets to each of the center coordinates
    # to reveal the global position of the bounding boxes.
    grid = np.arange(grid_size)
    x_s, y_s = np.meshgrid(grid, grid)

    x_offset = torch.FloatTensor(x_s).view(-1, 1)
    y_offset = torch.FloatTensor(y_s).view(-1, 1)

    if gpu:
        x_offset = x_offset.cuda()
        y_offset = y_offset.cuda()

    x_y_offset = torch.cat((x_offset, y_offset), 1).repeat(1, num_anchors).view(-1, 2).unsqueeze(0)
    predictions[:, :, :2] += x_y_offset  # offset the coordinates

    # Log transform the height and width of the bounding box using the formula:
    # new_dimension = anchor_dimension * exp(old_dimension)
    anchors = torch.FloatTensor(anchors)

    if gpu:
        anchors = anchors.cuda()

    anchors = anchors.repeat(grid_size * grid_size, 1).unsqueeze(0)
    predictions[:, :, 2:4] = torch.exp(predictions[:, :, 2:4]) * anchors  # log transform the dimensions

    # Now scale back everything to the size of the input image
    predictions[:, :, :4] *= stride

    return predictions


# Test code
if __name__ == '__main__':
    blocks = parse_cfg("./cfg/yolov3-voc.cfg")
    network_info, module_list = create_network(blocks)
    print(len(blocks), len(module_list))

