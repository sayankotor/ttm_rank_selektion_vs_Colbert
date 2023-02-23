import numpy as np
import torch
from torch import nn
import tntorch as tn

#import tensorly as tl
#from tensorly.decomposition import tensor_train
#tl.set_backend('pytorch')

import tntorch as tn



def matrix_to_tt_cores(matrix, shapes, ranks):
  shapes = np.asarray(shapes)
  matrix = matrix.reshape(list(shapes.flatten()))
  print (matrix.shape)
  d = len(shapes[0])
  transpose_idx = list(np.arange(2 * d).reshape(2, d).T.flatten())
  matrix = matrix.permute(*transpose_idx)
  print (matrix.shape)
  newshape = np.prod(shapes, 0)
  matrix = matrix.reshape(list(newshape))
  print (matrix.shape)
  #tt = tn.Tensor(matrix, ranks_tt=ranks)
  tt = tensor_train(full, rank = ranks)



  newcores = []
  for core, s1, s2, r1, r2 in zip(tt.cores,
                                  shapes[0], shapes[1],
                                  tt.ranks_tt, tt.ranks_tt[1:]):
    newcores.append(core.reshape((r1, s1, s2, r2)))
  return newcores


def ttmatmul(cores, t, shapes, ranks):
  ranks = [1] + ranks + [1]
  tshape = t.shape

  t = t.transpose(1, 0)
  t = t.reshape((-1, shapes[1][-1], 1))
  ndims = len(cores)
  for i in reversed(range(ndims)):
    t = torch.einsum('aijb,rjb->ira', (cores[i], t))
    if i:
      t = t.reshape((-1, shapes[1][i - 1], ranks[i]))
  t = t.reshape((int(np.prod(shapes[0])), tshape[1]))
  return t


def transpose(cores):
    result = []
    for c in cores:
        result.append(c.permute((0, 2, 1, 3)))
    return result


def matmultt(t, cores, shapes, ranks):
    #t = t.transpose(1, 0)
    #cores = transpose(cores)
    shapes = [shapes[1], shapes[0]]
    return ttmatmul(cores, t, shapes, ranks).transpose(1, 0)


def tt_multiply_tt_custom(cores: nn.ParameterList, vector : torch.Tensor, shapes, in_dims, ranks):
    """
    tensor: bs*768
    in_dims = [32, 24]
    """
    
    #print ("shapes, in_dims, ranks", shapes, in_dims, ranks)
    
    bs = vector.shape[0]
    seq_len = vector.shape[1]
    print (bs, seq_len)
    result = vector.reshape(bs*seq_len, in_dims[0], -1)
    core = cores[0].reshape(in_dims[0], cores[0].shape[2])
    result = torch.einsum('bid,ir->bdr', result, core)
    for i in range(1, len(cores)):
        if (i < len(in_dims)):
            result = result.reshape(-1, in_dims[i], cores[i].shape[0])
            core = cores[i].reshape(cores[i].shape[0], in_dims[i], -1, cores[i].shape[2])
            result = torch.einsum('bdr,rdac->bac', result, core)
        else:
            result = result.reshape(bs, -1, cores[i].shape[0])
            core = cores[i].reshape(cores[i].shape[0], -1, cores[i].shape[2])
            result = torch.einsum('bdr,rga->bdga', result, core)
    return result.reshape(bs, seq_len, -1)
            
    
class TTLayer(nn.Module):
    def __init__(self, layer, ranks, shapes, in_dims):
        super(TTLayer, self).__init__()
        self.ranks = ranks
        self.in_dims = in_dims
        self.shapes = shapes
        print ("layer.shape", layer.weight.shape)
        with torch.no_grad():
            #weight = layer.weight.transpose(1, 0)
            self.cores = nn.ParameterList(
            map(nn.Parameter, tn.Tensor(layer.weight.reshape(self.shapes), ranks_tt=ranks[1 : len(ranks)-1]).cores))
        self.bias = layer.bias

    def forward(self, inputs):
        out = tt_multiply_tt_custom(self.cores, inputs, self.shapes, self.in_dims, self.ranks)
        out = out + self.bias
        return out
