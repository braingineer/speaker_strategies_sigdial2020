import torch


class MLPEncoder(torch.nn.Sequential):
    """
    A simple Multilayer Perceptron (MLP) Encoder.

    The MLP is a series of Linear modules interspersed with the nonlinearity
    specified in the input arguments.  

    Args:
        size_in (int): the input size
        layer_sizes (tuple, list): a set of sizes to use in the MLP
        nonlinearity (torch.nn.activation.*): [default=torch.nn.ELU] 
            A nonlinearity used in between each layer of the MLP
        add_final_nonlinearity (bool): optinally add the last linearity; useful for MLPs meant to be used
            in a softmax.
    """
    def __init__(self, size_in, layer_sizes, nonlinearity=torch.nn.ELU, add_final_nonlinearity=True):
        super(MLPEncoder, self).__init__()
        for i, layer_size in enumerate(layer_sizes):
            self.add_module(f"fc{i}", torch.nn.Linear(size_in, layer_size))
            if i + 1 < len(layer_sizes) or add_final_nonlinearity:
                self.add_module(f"nonlinearity{i}", nonlinearity())
            size_in = layer_size
        self.in_features = size_in
        self.out_features = layer_sizes[-1]