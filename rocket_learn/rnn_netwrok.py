import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from typing import Union, Tuple

str_to_layer_fn = {
    "fc": nn.Linear,
    "rnn": nn.LSTMCell,
}

#mostly ripped from https://github.com/npitsillos/deepRLalgos
def create_fn(input_dim: Tuple[int], layers: Union[Tuple[int, int], Tuple[Tuple[str, int]]],
              layer_fn: Union[nn.Linear, nn.Conv2d, nn.LSTM] = None) -> nn.ModuleList():
    """
        Creates a ModuleList torch module containing the layers in the base network.
        :param input_dim: A tuple of ints defining the input size.
        :param layers: A tuple defining the layer sizes if a single layer type
            and the type of layer and their size if a custom model is needed.
        :param layer_fn: Type of layer if a single type, None if a custom
            model is needed.
        :return: nn.ModuleList containing the layers of the base network.
    """
    net = nn.ModuleList()
    if layer_fn is not None:
        # This is the case where a single type of layer policy is needed
        net += [layer_fn(*input_dim, layers[0])]
        for layer in range(len(layers) - 1):
            net += [layer_fn(layers[layer], layers[layer + 1])]
    else:
        net += [str_to_layer_fn[layers[0][0]](*input_dim, layers[0][1])]
        for layer in range(len(layers) - 1):
            net += [str_to_layer_fn[layers[layer + 1][0]](layers[layer][1], layers[layer + 1][1])]
    return net

