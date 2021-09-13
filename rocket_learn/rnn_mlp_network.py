import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from typing import Union, Tuple

str_to_layer_fn = {
    "fc": nn.Linear,
    "rnn": nn.LSTMCell,
}


def to_tensor(np_array, device="cpu"):
    if isinstance(np_array, np.ndarray):
        return torch.from_numpy(np_array).float().to(device)
    return np_array.float().to(device)


# mostly ripped from https://github.com/npitsillos/deepRLalgos
# need to heavily test this against sb2 ppo lstm policy baseline
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


class Network(nn.Module):
    """
        Wraps torch nn.Module to provide device type tracking.
    """

    def __init__(self):
        super().__init__()
        self.device = "cpu"

    def to(self, *args, **kwargs):
        """
            Override super method to track device.
        """
        if kwargs.get("device") is not None:
            self.device = kwargs.get("device")
        elif isinstance(args[0], str):
            self.device = args[0]
        return super().to(*args, **kwargs)

    def cuda(self, device=0):
        """
            Override super method to track device.
        """
        self.device = "cuda:" + str(device)
        return super().cuda(device)


class BaseNet(Network):
    """
        Base class which all policies inherit from.
    """

    def __init__(self, latent_size):
        super().__init__()
        self.latent_size = latent_size
        self.is_recurrent = False

    def set_layer_attrs(self):
        for i, layer in enumerate(self.layers):
            if isinstance(layer, nn.Linear):
                name = "fc"
            else:
                name = "rnn"
            self.__setattr__(f"{name}{i}", layer)


class RecurrentBase(BaseNet):
    """
        Base class for all networks using only recurrent layers.
        :param input_dim: A tuple of ints defining the input size.
        :param layers: A tuple defining the layer sizes if a single layer type
            and the type of layer and their size if a custom model is needed.
    """

    def __init__(self, input_dim, layers):
        super().__init__(layers[-1])

        self.layers = create_fn(input_dim, layers, nn.LSTMCell)
        self.is_recurrent = True
        self.set_layer_attrs()
        self.hidden = None
        self.cell = None

    def init_lstm_state(self, batch_size=1):
        # Do we also want to init from buffer?
        self.hidden = [torch.zeros(batch_size, layer.hidden_size) for layer in self.layers]
        self.cell = [torch.zeros(batch_size, layer.hidden_size) for layer in self.layers]

    def forward(self, x):
        # Two cases here either 3-dim x -> T x B x F, batch of trajectories having T timesteps
        # or 2-dim/1-dim -> batch of timesteps
        if len(x.shape) == 3:  # batch of trajectories
            self.init_lstm_state(x.size(1))

            x_s = []
            for x_t in x:
                for idx, layer in enumerate(self.layers):
                    hidden, cell = self.hidden[idx], self.cell[idx]
                    self.hidden[idx], self.cell[idx] = layer(x_t, (hidden, cell))
                    x_t = self.hidden[idx]
                x_s.append(x_t)

            x = torch.stack(x_s)
        else:
            dim = len(x.shape)
            if dim == 1:
                x = x.view(1, -1)

            for idx, layer in enumerate(self.layers):
                hidden, cell = self.hidden[idx], self.cell[idx]
                self.hidden[idx], self.cell[idx] = layer(x, (hidden, cell))
                x = self.hidden[idx]

            if dim == 1:
                x = x.view(-1)

        return x
