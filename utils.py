from __future__ import division

from constant import *
from layers import EmptyLayer

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


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
            block["type"] = line[1:-1].rstrip()
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
    assert net_info[LPROP_TYPE] != LAYER_NET, "Could not find the locate the network info layer. " \
                                              "Please check the cfg file."

    module_list = nn.ModuleList()
    prev_filters = 3  # For the 3 channels of an RGB image
    output_filters = []  # to keep track of the output dimensions of each layer

    # Begin the processing of the remaining blocks of the network.
    # Check the type of the module and create a pytorch implementation of it.
    # Append the layer to the module_list
    for index, block in enumerate(blocks[1:]):
        module = nn.Sequential()  # for grouping layers such as conv, activation and batch norm

        if block[LPROP_TYPE] == LAYER_CONVOLUTIONAL:
            activation_type = block[LPROP_ACTIVATION]
            filters= int(block[LPROP_FILTERS])
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

            # Positive anotation
            if start > 0:
                start = start - index
            if end > 0:
                end = end - index

            route = EmptyLayer()
            module.add_module("route_{0}".format(index), route)

            if end < 0:
                filters = output_filters[index + start] + output_filters[index + end]
            else:
                filters= output_filters[index + start]

