"""
Implementation of NICE bijective triangular-jacobian layers.
"""
import ipdb as pdb
import torch
import torch.nn as nn
import numpy as np


# ===== ===== Coupling Layer Implementations ===== =====

def _get_even(xs): return xs[:, 0::2]


def _get_odd(xs): return xs[:, 1::2]


def _interleave(first, second, order):
    """
    Given 2 rank-2 tensors with same batch dimension, interleave their columns.

    The tensors "first" and "second" are assumed to be of shape (B,M) and (B,N)
    where M = N or N+1, repsectively.

    'order' indicates columns to go through the activation function
    """
    cols = []  # len=784
    if order == 'even':
        for k in range(second.shape[1]):
            cols.append(first[:, k])  # (16,392)
            cols.append(second[:, k])  # (16,392)
        if first.shape[1] > second.shape[1]:
            cols.append(first[:, -1])
            # 길이가 하나 더 많으면, 마지막column을 복사붙여넣기
    else:
        for k in range(first.shape[1]):
            cols.append(second[:, k])
            cols.append(first[:, k])
        if second.shape[1] > first.shape[1]:
            cols.append(second[:, -1])
    return torch.stack(cols, dim=1)  # (16,784), 엮어진 메트릭스


class _BaseCouplingLayer(nn.Module):
    def __init__(self, dim, partition, nonlinearity):
        """
        Base coupling layer that handles the permutation of the inputs and wraps
        an instance of torch.nn.Module. <== IMPORTANT!

        Usage:
        >> layer = AdditiveCouplingLayer(1000, 'even', nn.Sequential(...))

        Args:
        * dim: dimension of the inputs.
        * partition: str, 'even' or 'odd'. If 'even', the even-valued columns are sent to
        pass through the activation module.
        * nonlinearity: an instance of torch.nn.Module.
        """
        super(_BaseCouplingLayer, self).__init__()
        # store input dimension of incoming values:
        self.dim = dim
        # store partition choice and make shorthands for 1st and second partitions:
        assert (partition in [
                'even', 'odd']), "[_BaseCouplingLayer] Partition type must be `even` or `odd`!"
        self.partition = partition
        if (partition == 'even'):
            self._first = _get_even
            self._second = _get_odd
        else:
            self._first = _get_odd
            self._second = _get_even
        # store nonlinear function module:
        # (n.b. this can be a complex instance of torch.nn.Module, for ex. a deep ReLU network)
        # torch.nn.Module의 기능에 customizing을 한 것
        self.add_module('nonlinearity', nonlinearity)

    def forward(self, x):
        """Map an input through the partition and nonlinearity."""
        return _interleave(
            self._first(x),
            self.coupling_law(self._second(
                x), self.nonlinearity(self._first(x))),
            self.partition
        )

    def inverse(self, y):
        """Inverse mapping through the layer. Gradients should be turned off for this pass."""
        return _interleave(
            self._first(y),
            self.anticoupling_law(self._second(
                y), self.nonlinearity(self._first(y))),
            self.partition
        )

    def coupling_law(self, a, b):
        # (a,b) --> g(a,b)
        raise NotImplementedError(
            "[_BaseCouplingLayer] Don't call abstract base layer!")

    def anticoupling_law(self, a, b):
        # (a,b) --> g^{-1}(a,b)
        raise NotImplementedError(
            "[_BaseCouplingLayer] Don't call abstract base layer!")


class AdditiveCouplingLayer(_BaseCouplingLayer):
    """Layer with coupling law g(a;b) := a + b."""

    def coupling_law(self, a, b):
        return (a + b)

    def anticoupling_law(self, a, b):
        return (a - b)


class MultiplicativeCouplingLayer(_BaseCouplingLayer):
    """Layer with coupling law g(a;b) := a .* b."""

    def coupling_law(self, a, b):
        return torch.mul(a, b)

    def anticoupling_law(self, a, b):
        return torch.mul(a, torch.reciprocal(b))


class AffineCouplingLayer(_BaseCouplingLayer):
    """Layer with coupling law g(a;b) := a .* b1 + b2, where (b1,b2) is a partition of b."""

    def coupling_law(self, a, b):
        return torch.mul(a, self._first(b)) + self._second(b)

    def anticoupling_law(self, a, b):
        # TODO
        raise NotImplementedError("TODO: AffineCouplingLayer (sorry!)")
