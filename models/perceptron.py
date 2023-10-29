from torch import nn
import torch

class ModelPerceptron(nn.Module):
    def __init__(
        self,
        dim_input: int,
        dim_output: int,
        dim_hidden: int,
        n_hidden_layers: int,
        leaky_relu_coef: float = 0.15,
        is_bias: bool = True,
    ):
        """Instatiate ModelPerceptron

        :param dim_input: dimension of input layer
        :type dim_input: int
        :param dim_output: dimension of output layer
        :type dim_output: int
        :param dim_hidden: dimension of hidden layers
        :type dim_hidden: int
        :param n_hidden_layers: number of hidden layers
        :type n_hidden_layers: int
        :param leaky_relu_coef: coefficient for leaky_relu activation functions, defaults to 0.15
        :type leaky_relu_coef: float, optional
        :param is_bias: whether to use bias in linear layers, defaults to True
        :type is_bias: bool, optional
        """
        super().__init__()
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.n_hidden_layers = n_hidden_layers
        self.leaky_relu_coef = leaky_relu_coef
        self.is_bias = is_bias

        self.input_layer = nn.Linear(dim_input, dim_hidden, bias=is_bias)
        self.hidden_layers = nn.ModuleList(
            [
                nn.Linear(dim_hidden, dim_hidden, bias=is_bias)
                for _ in range(n_hidden_layers)
            ]
        )
        self.output_layer = nn.Linear(dim_hidden, dim_output, bias=is_bias)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        """Make forward pass through the perceptron

        :param x: input Float Tensor
        :type x: torch.FloatTensor
        :return: output of perceptron
        :rtype: torch.FloatTensor
        """
        x = nn.functional.leaky_relu(
            self.input_layer(x), negative_slope=self.leaky_relu_coef
        )
        for layer in self.hidden_layers:
            x = nn.functional.leaky_relu(layer(x), negative_slope=self.leaky_relu_coef)
        x = self.output_layer(x)
        return x