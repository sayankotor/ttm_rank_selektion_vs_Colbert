from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch as T
from opt_einsum import contract_expression
from opt_einsum.contract import ContractExpression

from .linalg_tt import ttd

__all__ = ('CompressedLinear', 'TTCompressedLinear')


def factorize(value: int) -> Dict[int, int]:
    """Function factorize factorizes an interger on prime numbers.

    :param value: Interger number to factorize.
    :return: A mapping from factor to its multiplicity.
    """
    primes = {}

    def exhaust(value: int, prime: int) -> int:
        """Divide :value: by :prime: until it is possible.
        """
        count = 0
        while value % prime == 0:
            value //= prime
            count += 1
        if count:
            primes[prime] = count
        return value

    # There is no primes for such numbers.
    if value < 2:
        return primes

    # Find all primes 2 in the number.
    value = exhaust(value, prime=2)

    # Now we can try all numbers from 3 to sqrt with stride 2.
    prime = 3
    while prime**2 <= value:
        value = exhaust(value, prime)
        prime += 2

    # If a remained is not lesser than 3 than it means that this number a prime
    # itself.
    if value > 2:
        primes[value] = 1

    return primes


def make_contraction(shape, rank, batch_size=32,
                     seqlen=512) -> ContractExpression:
    ndim = len(rank) - 1
    row_shape, col_shape = shape

    # Generate all contraction indexes.
    row_ix, col_ix = np.arange(2 * ndim).reshape(2, ndim)
    rank_ix = 2 * ndim + np.arange(ndim + 1)
    batch_ix = 4 * ndim  # Zero-based index.

    # Order indexes of cores.
    cores_ix = np.column_stack([rank_ix[:-1], row_ix, col_ix, rank_ix[1:]])
    cores_shape = zip(rank[:-1], row_shape, col_shape, rank[1:])

    # Order indexes of input (contraction by columns: X G_1 G_2 ... G_d).
    input_ix = np.insert(row_ix, 0, batch_ix)
    input_shape = (batch_size * seqlen, ) + row_shape

    # Order indexes of output (append rank indexes as well).
    output_ix = np.insert(col_ix, 0, batch_ix)
    output_ix = np.append(output_ix, (rank_ix[0], rank_ix[-1]))

    # Prepare contraction operands.
    ops = [input_shape, input_ix]
    for core_ix, core_shape in zip(cores_ix, cores_shape):
        ops.append(core_shape)
        ops.append(core_ix)
    ops.append(output_ix)
    ops = [tuple(op) for op in ops]

    return contract_expression(*ops)


class TTCompressedLinear(CompressedLinear):
    """Class TTCompressedLinear is a layer which represents a weight matrix of
    linear layer in factorized view as tensor train matrix.

    >>> linear_layer = T.nn.Linear(6, 6)
    >>> tt_layer = TTCompressedLinear \
    ...     .from_linear(linear_layer, rank=2, shape=((2, 3), (3, 2)))
    """

    def __init__(self, cores: Sequence[T.Tensor],
                 bias: Optional[T.Tensor] = None):
        super().__init__()

        for i, core in enumerate(cores):
            if core.ndim != 4:
                raise ValueError('Expected number of dimensions of the '
                                 f'{i}-th core is 4 but given {cores.ndim}.')

        # Prepare contaction expression.
        self.rank = (1, ) + tuple(core.shape[3] for core in cores)
        self.shape = (tuple(core.shape[1] for core in cores),
                      tuple(core.shape[2] for core in cores))
        self.contact = make_contraction(self.shape, self.rank)

        # TT-matrix is applied on the left. So, this defines number of input
        # and output features.
        self.in_features = np.prod(self.shape[0])
        self.out_features = np.prod(self.shape[1])

        # Create trainable variables.
        self.cores = T.nn.ParameterList(T.nn.Parameter(core) for core in cores)
        self.bias = None
        if bias is not None:
            if bias.size() != self.out_features:
                raise ValueError(f'Expected bias size is {self.out_features} '
                                 f'but its shape is {bias.shape}.')
            self.bias = T.nn.Parameter(bias)

    def forward(self, input: T.Tensor) -> T.Tensor:
        # We need replace the feature dimension with multi-dimension to contact
        # with TT-matrix.
        input_shape = input.shape
        input = input.reshape(-1, *self.shape[0])

        # Contract input with weights and replace back multi-dimension with
        # feature dimension.
        output = self.contact(input, *self.cores)
        output = output.reshape(*input_shape[:-1], self.out_features)

        if self.bias is not None:
            output += self.bias
        return output

    @classmethod
    def from_linear(cls, linear: T.nn.Linear,
                    shape: Tuple[Tuple[int], Tuple[int]], rank: int, **kwargs):
        ndim = len(shape[0])

        # Prepare information about shape and rank of TT (not TTM).
        tt_rank = (1, ) + (rank, ) * (ndim - 1) + (1, )
        tt_shape = tuple(n * m for n, m in zip(*shape))

        # Reshape weight matrix to tensor indexes like TT-matrix.
        matrix = linear.weight.data.T
        tensor = matrix.reshape(shape[0] + shape[1])
        for i in range(ndim - 1):
            tensor = tensor.moveaxis(ndim + i, 2 * i + 1)

        # Reshape TT-matrix to a plain TT and apply decomposition.
        tensor = tensor.reshape(tt_shape)
        cores = ttd(tensor, tt_rank, **kwargs)

        # Reshape TT-cores back to TT-matrix cores (TTM-cores).
        core_shapes = zip(tt_rank, *shape, tt_rank[1:])
        cores = [core.reshape(core_shape)
                 for core, core_shape in zip(cores, core_shapes)]

        # Make copy of bias if it exists.
        bias = None
        if linear.bias is not None:
            bias = T.clone(linear.bias.data)

        return TTCompressedLinear(cores, bias)
