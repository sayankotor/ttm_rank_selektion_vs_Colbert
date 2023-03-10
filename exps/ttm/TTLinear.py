from typing import List, Callable
import torch
import torch.nn as nn
import tntorch as tn
from math import sqrt
from .ttmatrix_fisher import TTMatrix

from .forward_backward import forward, einsum_forward

class TTLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, ranks: List[int], input_dims: List[int],
                 output_dims: List[int], bias: bool = True, device=None, dtype=None,
                 forward_fn: Callable = einsum_forward):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.ranks = list(ranks)
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.forward_fn = forward_fn

        # Initialize weights from uniform[-1 / sqrt(in_features), 1 / sqrt(in_features)]
        factory_kwargs = {"device": device, "dtype": dtype}
        init = torch.rand(in_features, out_features, **factory_kwargs)
        init = (2 * init - 1) / sqrt(in_features)
        
        init_fisher = torch.zeros(init.shape)
        for i in range(0, min(init.shape)):
            init_fisher[i, i] = 1.0

        
        weight = TTMatrix(init, list(ranks), input_dims, output_dims)

        # torch doesn't recognize attributes of self.weight as parameters,
        # so we have to use ParameterList
        self.cores = nn.ParameterList([nn.Parameter(core) for core in weight.cores])
        
        #self.bias = nn.Parameter(torch.zeros(out_features)) if bias is not None else None


        if bias:
            init = torch.rand(out_features, **factory_kwargs)
            init = (2 * init - 1) / sqrt(out_features)
            self.bias = nn.Parameter(init)
        else:
            self.register_parameter('bias', None)


    def forward(self, x: torch.Tensor):
        res = self.forward_fn(self.cores, x)

        new_shape = x.shape[:-1] + (self.out_features,)
        res = res.reshape(*new_shape)

        if self.bias is not None:
            res += self.bias

        return res

    def set_weight(self, new_weights: torch.Tensor, need_singular_values: bool):
        # in regular linear layer weights are transposed, so we transpose back
        new_weights = new_weights.clone().detach().T

        shape = torch.Size((self.in_features, self.out_features))
        assert new_weights.shape == shape, f"Expected shape {shape}, got {new_weights.shape}"

        weight = TTMatrix(new_weights, self.ranks, self.input_dims, self.output_dims, is_from_weight = need_singular_values)
        self.cores = nn.ParameterList([nn.Parameter(core) for core in weight.cores])
        weight.cores = self.cores
        
    def set_bias(self, new_bias:torch.Tensor):
        self.bias = new_bias
        
    def set_weight_with_fisher(self, new_weights: torch.Tensor, fisher_matrix: torch.Tensor):
        # in regular linear layer weights are transposed, so we transpose back
        new_weights = new_weights.clone().detach().T
        fisher_matrix = fisher_matrix.clone().detach().T

        shape = torch.Size((self.in_features, self.out_features))
        assert new_weights.shape == shape, f"Expected shape {shape}, got {new_weights.shape}"

        weight = TTMatrix(new_weights, fisher_matrix, self.ranks, self.input_dims, self.output_dims)
        self.cores = nn.ParameterList([nn.Parameter(core) for core in weight.cores])
        weight.cores = self.cores

    def set_from_linear(self, linear: nn.Linear):
        self.set_weight_with_fisher(linear.weight.data, torch.diag(torch.ones(linear.weight.data.shape)))
        self.bias = nn.Parameter(linear.bias.data.clone()) if linear.bias is not None else None
