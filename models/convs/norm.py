# Batch Renormalization for convolutional neural nets (2D) implementation based
# on https://arxiv.org/abs/1702.03275
# https://github.com/ludvb/batchrenorm/blob/master/batchrenorm/batchrenorm.py


__all__ = ["BatchRenorm1d", "BatchRenorm2d", "BatchRenorm3d"]

from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.parameter import Parameter
from torch.nn import functional as F
import torch
import torch.nn as nn

class Norm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(Norm2d, self).__init__()

        self.n_feats = num_features
        self.n_updates = n_updates
        self.bn = nn.BatchNorm2d(num_features, eps=eps, momentum=momentum)
        self.register_buffer(
            "num_batches_tracked", torch.tensor(0, dtype=torch.long)
        )
        # Registering identity matrix in buffer makes easy to manage in memory
        # It automatically handles cpu or gpu memory usage
        # self.register_buffer(
        #     'eye', torch.eye(self.n_feats, dtype=torch.float).view(self.n_feats, self.n_feats, 1, 1)
        # )
        # self.norm = nn.Conv2d(n_channels, n_channels, 1, 1, 0)

        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x, inverse=False):
        if self.training and self.num_batches_tracked < self.n_updates:
            # print("training bn")
            x = self.bn(x)
            self.num_batches_tracked += 1

        # mean = self.bn.running_mean
        # std = torch.sqrt(self.bn.running_var)

        # # Calculate conv2d with statistics
        # weight = self.eye / std.view(self.n_feats, 1, 1, 1)
        # bias = -mean / std

        # x = F.conv2d(x, weight, bias=bias)
        # return x

        
        if x.dim() > 2:
            x = x.transpose(1, -1)

        if not inverse:
            # print("forwarding")
            # print("self.bn.running_mean:", self.bn.running_mean)
            # print("self.bn.running_var:", self.bn.running_var)
            x = (x - self.bn.running_mean) / torch.sqrt(self.bn.running_var)
        else:
            # print("inverse")
            # print("self.bn.running_mean:", self.bn.running_mean)
            # print("self.bn.running_var:", self.bn.running_var)
            x = x * torch.sqrt(self.bn.running_var) + self.bn.running_mean

        if x.dim() > 2:
            x = x.transpose(1, -1)
        return x


class _Norm(_BatchNorm):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, n_updates=10000):
        super(_Norm, self).__init__(num_features, eps, momentum, affine)

        self.register_buffer(
            "num_batches_tracked", torch.tensor(0, dtype=torch.long)
        )
        self.n_updates = n_updates

        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x, update_stat=False, inverse=False):
        self._check_input_dim(x)

        # Update running_mean and running_std by using torch.functional.norm
        if update_stat and self.num_batches_tracked < self.n_updates:
            bn = F.batch_norm(
                x, self.running_mean, self.running_var, weight=None, bias=self.bias,
                training=self.training, momentum=self.momentum, eps=self.eps)

            self.num_batches_tracked += 1

        if x.dim() > 2:
            x = x.transpose(1, -1)
        
        # print("self.running_mean:", self.running_mean)
        # print("self.running_var:", self.running_var)

        # print("x.shape:", x.shape)
        if not inverse:
            x = (x - self.running_mean) / torch.sqrt(self.running_var)
        else:
            x = x * torch.sqrt(self.running_var) + self.running_mean

        if x.dim() > 2:
            x = x.transpose(1, -1)
        return x

class Norm1d_(_Norm):
    def _check_input_dim(self, input):
        if input.dim() != 2 and input.dim() != 3:
            raise ValueError('expected 2D or 3D input (got {}D input)'.format(input.dim()))


class Norm2d_(_Norm):
    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'.format(input.dim()))


class Norm3d_(_Norm):
    def _check_input_dim(self, input):
        if input.dim() != 5:
            raise ValueError('expected 5D input (got {}D input)'.format(input.dim()))



# class _Norm(torch.jit.ScriptModule):
#     def __init__(
#         self,
#         num_features: int,
#         eps: float = 1e-3,
#         momentum: float = 0.01,
#         affine: bool = True,
#     ):
#         super().__init__()
#         self.register_buffer(
#             "running_mean", torch.zeros(num_features, dtype=torch.float)
#         )
#         self.register_buffer(
#             "running_std", torch.ones(num_features, dtype=torch.float)
#         )
#         self.register_buffer(
#             "num_batches_tracked", torch.tensor(0, dtype=torch.long)
#         )
#         self.weight = torch.nn.Parameter(
#             torch.ones(num_features, dtype=torch.float)
#         )
#         self.bias = torch.nn.Parameter(
#             torch.zeros(num_features, dtype=torch.float)
#         )
#         self.eps = eps
#         self.step = 0
#         self.momentum = momentum

#     def _check_input_dim(self, x: torch.Tensor) -> None:
#         raise NotImplementedError()  # pragma: no cover

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         self._check_input_dim(x)
#         if x.dim() > 2:
#             x = x.transpose(1, -1)
#         if self.training:
#             dims = [i for i in range(x.dim() - 1)]
#             batch_mean = x.mean(dims)
#             batch_std = x.std(dims, unbiased=False) + self.eps
            
#             self.running_mean += self.momentum * (
#                 batch_mean.detach() - self.running_mean
#             )
#             self.running_std += self.momentum * (
#                 batch_std.detach() - self.running_std
#             )
#             self.num_batches_tracked += 1
#         x = (x - self.running_mean) / self.running_std
        
#         if x.dim() > 2:
#             x = x.transpose(1, -1)
#         return x

#     def inverse(self, x: torch.Tensor) -> torch.Tensor:
#         self._check_input_dim(x)
#         if x.dim() > 2:
#             x = x.transpose(1, -1)

#         x = x * self.running_std + self.running_mean
        
#         if x.dim() > 2:
#             x = x.transpose(1, -1)
#         return x

# class Norm1d(_Norm):
#     def _check_input_dim(self, x: torch.Tensor) -> None:
#         if x.dim() not in [2, 3]:
#             raise ValueError("expected 2D or 3D input (got {x.dim()}D input)")


# class Norm2d(_Norm):
#     def _check_input_dim(self, x: torch.Tensor) -> None:
#         if x.dim() != 4:
#             raise ValueError("expected 4D input (got {x.dim()}D input)")


# class Norm3d(_Norm):
#     def _check_input_dim(self, x: torch.Tensor) -> None:
#         if x.dim() != 5:
#             raise ValueError("expected 5D input (got {x.dim()}D input)")