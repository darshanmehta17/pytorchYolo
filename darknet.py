import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from constants import *
from torch.autograd import Variable
from utils import parse_cfg, create_network, transform_predictions


class Darknet(nn.Module):
    """
    # TODO: Fill in documentation
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


if __name__ == '__main__':
    model = Darknet("./cfg/yolov3-voc.cfg")
    inp = get_test_input("./data/images_subset/2018_12_04_11_31_49.64.jpg", dim=416)
    pred = model(inp, False)
    print(pred)
