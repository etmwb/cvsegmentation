##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Hang Zhang
## Email: zhanghang0704@gmail.com
## Copyright (c) 2018
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

"""Synchronized Cross-GPU Batch Normalization functions"""
import warnings
try:
    from queue import Queue
except ImportError:
    from Queue import Queue

import torch
import torch.cuda.comm as comm
from torch.autograd import Function
from torch.nn.modules.batchnorm import _BatchNorm
from torch.autograd.function import once_differentiable
from . import encoding_ext
from .dist_syncbn import dist_syncbatchnorm

class moments_(Function):
    @staticmethod
    def forward(ctx, x):
        ex, ex2 = encoding_ext.expectation_forward(x)
        ctx.save_for_backward(x)
        return ex, ex2

    @staticmethod
    def backward(ctx, dex, dex2):
        x, = ctx.saved_tensors
        dx = encoding_ext.expectation_backward(x, dex, dex2)
        return dx

class syncbatchnorm_(Function):
    @classmethod
    def forward(cls, ctx, x, gamma, beta, running_mean, running_var,
                extra, sync=True, training=True, momentum=0.1, eps=1e-05,
                activation="none", slope=0.01):
        # save context
        cls._parse_extra(ctx, extra)
        ctx.sync = sync
        ctx.training = training
        ctx.momentum = momentum
        ctx.eps = eps
        ctx.activation = activation
        ctx.slope = slope
        assert activation == 'none'

        # continous inputs
        x = x.contiguous()
        gamma = gamma.contiguous()
        beta = beta.contiguous()

        if ctx.training:
            _ex, _exs = encoding_ext.expectation_forward(x)

            if ctx.sync:
                if ctx.is_master:
                    _ex, _exs = [_ex.unsqueeze(0)], [_exs.unsqueeze(0)]
                    for _ in range(ctx.master_queue.maxsize):
                        _ex_w, _exs_w = ctx.master_queue.get()
                        ctx.master_queue.task_done()
                        _ex.append(_ex_w.unsqueeze(0))
                        _exs.append(_exs_w.unsqueeze(0))

                    _ex = comm.gather(_ex).mean(0)
                    _exs = comm.gather(_exs).mean(0)

                    tensors = comm.broadcast_coalesced((_ex, _exs), [_ex.get_device()] + ctx.worker_ids)
                    for ts, queue in zip(tensors[1:], ctx.worker_queues):
                        queue.put(ts)
                else:
                    ctx.master_queue.put((_ex, _exs))
                    _ex, _exs = ctx.worker_queue.get()
                    ctx.worker_queue.task_done()

            # Update running stats
            _var = _exs - _ex ** 2
            running_mean.mul_((1 - ctx.momentum)).add_(ctx.momentum * _ex)
            running_var.mul_((1 - ctx.momentum)).add_(ctx.momentum * _var)

            # Mark in-place modified tensors
            ctx.mark_dirty(running_mean, running_var)
        else:
            _ex, _var = running_mean.contiguous(), running_var.contiguous()
            _exs = _var + _ex ** 2 

        # BN forward + activation
        y = encoding_ext.batchnorm_forward(x, _ex, _exs, gamma, beta, ctx.eps)

        # Output
        ctx.save_for_backward(x, _ex, _exs, gamma, beta)
        return y

    @staticmethod
    @once_differentiable
    def backward(ctx, dz):
        x, _ex, _exs, gamma, beta = ctx.saved_tensors
        dz = dz.contiguous()

        # BN backward
        dx, _dex, _dexs, dgamma, dbeta = \
            encoding_ext.batchnorm_backward(dz, x, _ex, _exs, gamma, beta, ctx.eps)

        if ctx.training:
            if ctx.sync:
                if ctx.is_master:
                    _dex, _dexs = [_dex.unsqueeze(0)], [_dexs.unsqueeze(0)]
                    for _ in range(ctx.master_queue.maxsize):
                        _dex_w, _dexs_w = ctx.master_queue.get()
                        ctx.master_queue.task_done()
                        _dex.append(_dex_w.unsqueeze(0))
                        _dexs.append(_dexs_w.unsqueeze(0))

                    _dex = comm.gather(_dex).mean(0)
                    _dexs = comm.gather(_dexs).mean(0)

                    tensors = comm.broadcast_coalesced((_dex, _dexs), [_dex.get_device()] + ctx.worker_ids)
                    for ts, queue in zip(tensors[1:], ctx.worker_queues):
                        queue.put(ts)
                else:
                    ctx.master_queue.put((_dex, _dexs))
                    _dex, _dexs = ctx.worker_queue.get()
                    ctx.worker_queue.task_done()

            dx_ = encoding_ext.expectation_backward(x, _dex, _dexs)
            dx = dx + dx_

        return dx, dgamma, dbeta, None, None, None, None, None, None, None, None, None

    @staticmethod
    def _parse_extra(ctx, extra):
        ctx.is_master = extra["is_master"]
        if ctx.is_master:
            ctx.master_queue = extra["master_queue"]
            ctx.worker_queues = extra["worker_queues"]
            ctx.worker_ids = extra["worker_ids"]
        else:
            ctx.master_queue = extra["master_queue"]
            ctx.worker_queue = extra["worker_queue"]

def _act_forward(ctx, x):
    if ctx.activation.lower() == "leaky_relu":
        encoding_ext.leaky_relu_forward(x, ctx.slope)
    else:
        assert activation == 'none'

def _act_backward(ctx, x, dx):
    if ctx.activation.lower() == "leaky_relu":
        encoding_ext.leaky_relu_backward(x, dx, ctx.slope)
    else:
        assert activation == 'none'

class inp_syncbatchnorm_(Function):
    @classmethod
    def forward(cls, ctx, x, gamma, beta, running_mean, running_var,
                extra, sync=True, training=True, momentum=0.1, eps=1e-05,
                activation="none", slope=0.01):
        # save context
        cls._parse_extra(ctx, extra)
        ctx.sync = sync
        ctx.training = training
        ctx.momentum = momentum
        ctx.eps = eps
        ctx.activation = activation
        ctx.slope = slope

        # continous inputs
        x = x.contiguous()
        gamma = gamma.contiguous()
        beta = beta.contiguous()

        if ctx.training:
            _ex, _exs = encoding_ext.expectation_forward(x)

            if ctx.sync:
                if ctx.is_master:
                    _ex, _exs = [_ex.unsqueeze(0)], [_exs.unsqueeze(0)]
                    for _ in range(ctx.master_queue.maxsize):
                        _ex_w, _exs_w = ctx.master_queue.get()
                        ctx.master_queue.task_done()
                        _ex.append(_ex_w.unsqueeze(0))
                        _exs.append(_exs_w.unsqueeze(0))

                    _ex = comm.gather(_ex).mean(0)
                    _exs = comm.gather(_exs).mean(0)

                    tensors = comm.broadcast_coalesced((_ex, _exs), [_ex.get_device()] + ctx.worker_ids)
                    for ts, queue in zip(tensors[1:], ctx.worker_queues):
                        queue.put(ts)
                else:
                    ctx.master_queue.put((_ex, _exs))
                    _ex, _exs = ctx.worker_queue.get()
                    ctx.worker_queue.task_done()

            # Update running stats
            _var = _exs - _ex ** 2
            running_mean.mul_((1 - ctx.momentum)).add_(ctx.momentum * _ex)
            running_var.mul_((1 - ctx.momentum)).add_(ctx.momentum * _var)

            # Mark in-place modified tensors
            ctx.mark_dirty(x, running_mean, running_var)
        else:
            _ex, _var = running_mean.contiguous(), running_var.contiguous()
            _exs = _var + _ex ** 2 
            ctx.mark_dirty(x)

        # BN forward + activation
        encoding_ext.batchnorm_inp_forward(x, _ex, _exs, gamma, beta, ctx.eps)
        _act_forward(ctx, x)

        # Output
        ctx.save_for_backward(x, _ex, _exs, gamma, beta)
        return x

    @staticmethod
    @once_differentiable
    def backward(ctx, dz):
        z, _ex, _exs, gamma, beta = ctx.saved_tensors
        dz = dz.contiguous()

        # Undo activation
        _act_backward(ctx, z, dz)

        # BN backward
        dx, _dex, _dexs, dgamma, dbeta = \
            encoding_ext.batchnorm_inp_backward(dz, z, _ex, _exs, gamma, beta, ctx.eps)

        if ctx.training:
            if ctx.sync:
                if ctx.is_master:
                    _dex, _dexs = [_dex.unsqueeze(0)], [_dexs.unsqueeze(0)]
                    for _ in range(ctx.master_queue.maxsize):
                        _dex_w, _dexs_w = ctx.master_queue.get()
                        ctx.master_queue.task_done()
                        _dex.append(_dex_w.unsqueeze(0))
                        _dexs.append(_dexs_w.unsqueeze(0))

                    _dex = comm.gather(_dex).mean(0)
                    _dexs = comm.gather(_dexs).mean(0)

                    tensors = comm.broadcast_coalesced((_dex, _dexs), [_dex.get_device()] + ctx.worker_ids)
                    for ts, queue in zip(tensors[1:], ctx.worker_queues):
                        queue.put(ts)
                else:
                    ctx.master_queue.put((_dex, _dexs))
                    _dex, _dexs = ctx.worker_queue.get()
                    ctx.worker_queue.task_done()

            encoding_ext.expectation_inp_backward(dx, z, _dex, _dexs, _ex, _exs, gamma, beta, ctx.eps)

        return dx, dgamma, dbeta, None, None, None, None, None, None, None, None, None

    @staticmethod
    def _parse_extra(ctx, extra):
        ctx.is_master = extra["is_master"]
        if ctx.is_master:
            ctx.master_queue = extra["master_queue"]
            ctx.worker_queues = extra["worker_queues"]
            ctx.worker_ids = extra["worker_ids"]
        else:
            ctx.master_queue = extra["master_queue"]
            ctx.worker_queue = extra["worker_queue"]

moments = moments_.apply
syncbatchnorm = syncbatchnorm_.apply
inp_syncbatchnorm = inp_syncbatchnorm_.apply


class DistSyncBatchNorm(_BatchNorm):
    r"""Cross-GPU Synchronized Batch normalization (SyncBN)

    Standard BN [1]_ implementation only normalize the data within each device (GPU).
    SyncBN normalizes the input within the whole mini-batch.
    We follow the sync-onece implmentation described in the paper [2]_ .
    Please see the design idea in the `notes <./notes/syncbn.html>`_.

    .. math::

        y = \frac{x - mean[x]}{ \sqrt{Var[x] + \epsilon}} * gamma + beta

    The mean and standard-deviation are calculated per-channel over
    the mini-batches and gamma and beta are learnable parameter vectors
    of size C (where C is the input size).

    During training, this layer keeps a running estimate of its computed mean
    and variance. The running sum is kept with a default momentum of 0.1.

    During evaluation, this running mean/variance is used for normalization.

    Because the BatchNorm is done over the `C` dimension, computing statistics
    on `(N, H, W)` slices, it's common terminology to call this Spatial BatchNorm

    Args:
        num_features: num_features from an expected input of
            size batch_size x num_features x height x width
        eps: a value added to the denominator for numerical stability.
            Default: 1e-5
        momentum: the value used for the running_mean and running_var
            computation. Default: 0.1
        sync: a boolean value that when set to ``True``, synchronize across
            different gpus. Default: ``True``
        activation : str
            Name of the activation functions, one of: `leaky_relu` or `none`.
        slope : float
            Negative slope for the `leaky_relu` activation.

    Shape:
        - Input: :math:`(N, C, H, W)`
        - Output: :math:`(N, C, H, W)` (same shape as input)

    Reference:
        .. [1] Ioffe, Sergey, and Christian Szegedy. "Batch normalization: Accelerating deep network training by reducing internal covariate shift." *ICML 2015*
        .. [2] Hang Zhang, Kristin Dana, Jianping Shi, Zhongyue Zhang, Xiaogang Wang, Ambrish Tyagi, and Amit Agrawal. "Context Encoding for Semantic Segmentation." *CVPR 2018*

    Examples:
        >>> m = DistSyncBatchNorm(100)
        >>> net = torch.nn.parallel.DistributedDataParallel(m)
        >>> output = net(input)
    """
    def __init__(self, num_features, eps=1e-5, momentum=0.1, process_group=None):
        super(DistSyncBatchNorm, self).__init__(num_features, eps=eps, momentum=momentum, affine=True, track_running_stats=True)
        self.process_group = process_group

    def forward(self, x):
        need_sync = self.training or not self.track_running_stats
        process_group = None
        if need_sync:
            process_group = torch.distributed.group.WORLD
            if self.process_group:
                process_group = self.process_group
            world_size = torch.distributed.get_world_size(process_group)
            need_sync = world_size > 1

        # Resize the input to (B, C, -1).
        input_shape = x.size()
        x = x.view(input_shape[0], self.num_features, -1)
        #def forward(ctx, x, gamma, beta, running_mean, running_var, eps, momentum, training, process_group):
        y = dist_syncbatchnorm(x, self.weight, self.bias, self.running_mean, self.running_var,
                               self.eps, self.momentum, self.training, process_group)

        #_var = _exs - _ex ** 2
        #running_mean.mul_((1 - ctx.momentum)).add_(ctx.momentum * _ex)
        #running_var.mul_((1 - ctx.momentum)).add_(ctx.momentum * _var)
        return y.view(input_shape)

class SyncBatchNorm(_BatchNorm):
    r"""Cross-GPU Synchronized Batch normalization (SyncBN)

    Standard BN [1]_ implementation only normalize the data within each device (GPU).
    SyncBN normalizes the input within the whole mini-batch.
    We follow the sync-onece implmentation described in the paper [2]_ .
    Please see the design idea in the `notes <./notes/syncbn.html>`_.

    .. math::

        y = \frac{x - mean[x]}{ \sqrt{Var[x] + \epsilon}} * gamma + beta

    The mean and standard-deviation are calculated per-channel over
    the mini-batches and gamma and beta are learnable parameter vectors
    of size C (where C is the input size).

    During training, this layer keeps a running estimate of its computed mean
    and variance. The running sum is kept with a default momentum of 0.1.

    During evaluation, this running mean/variance is used for normalization.

    Because the BatchNorm is done over the `C` dimension, computing statistics
    on `(N, H, W)` slices, it's common terminology to call this Spatial BatchNorm

    Args:
        num_features: num_features from an expected input of
            size batch_size x num_features x height x width
        eps: a value added to the denominator for numerical stability.
            Default: 1e-5
        momentum: the value used for the running_mean and running_var
            computation. Default: 0.1
        sync: a boolean value that when set to ``True``, synchronize across
            different gpus. Default: ``True``
        activation : str
            Name of the activation functions, one of: `leaky_relu` or `none`.
        slope : float
            Negative slope for the `leaky_relu` activation.

    Shape:
        - Input: :math:`(N, C, H, W)`
        - Output: :math:`(N, C, H, W)` (same shape as input)

    Examples:
        >>> m = SyncBatchNorm(100)
        >>> net = torch.nn.DataParallel(m)
        >>> output = net(input)
        >>> # for Inpace ABN
        >>> ABN = partial(SyncBatchNorm, activation='leaky_relu', slope=0.01, sync=True, inplace=True)
    """

    def __init__(self, num_features, eps=1e-5, momentum=0.1, sync=True, activation="none", slope=0.01,
                 inplace=True):
        super(SyncBatchNorm, self).__init__(num_features, eps=eps, momentum=momentum, affine=True)
        self.activation = activation
        self.inplace = False if activation == 'none' else inplace
        #self.inplace = inplace
        self.slope = slope
        self.devices = list(range(torch.cuda.device_count()))
        self.sync = sync if len(self.devices) > 1 else False
        # Initialize queues
        self.worker_ids = self.devices[1:]
        self.master_queue = Queue(len(self.worker_ids))
        self.worker_queues = [Queue(1) for _ in self.worker_ids]
        # running_exs
        #self.register_buffer('running_exs', torch.ones(num_features))

    def _check_input_dim(self, x):
        pass

    def forward(self, x):
        if not self.training:
            return super().forward(x)
        # Resize the input to (B, C, -1).
        input_shape = x.size()
        x = x.view(input_shape[0], self.num_features, -1)
        if x.get_device() == self.devices[0]:
            # Master mode
            extra = {
                "is_master": True,
                "master_queue": self.master_queue,
                "worker_queues": self.worker_queues,
                "worker_ids": self.worker_ids
            }
        else:
            # Worker mode
            extra = {
                "is_master": False,
                "master_queue": self.master_queue,
                "worker_queue": self.worker_queues[self.worker_ids.index(x.get_device())]
            }
        if self.inplace:
            return inp_syncbatchnorm(x, self.weight, self.bias, self.running_mean, self.running_var,
                                     extra, self.sync, self.training, self.momentum, self.eps,
                                     self.activation, self.slope).view(input_shape)
        else:
            return syncbatchnorm(x, self.weight, self.bias, self.running_mean, self.running_var,
                                 extra, self.sync, self.training, self.momentum, self.eps,
                                 self.activation, self.slope).view(input_shape)

    def extra_repr(self):
        if self.activation == 'none':
            return 'sync={}'.format(self.sync)
        else:
            return 'sync={}, act={}, slope={}, inplace={}'.format(
                self.sync, self.activation, self.slope, self.inplace
            )

class BatchNorm1d(SyncBatchNorm):
    r"""
    .. warning::
        BatchNorm1d is deprecated in favor of :class:`encoding.nn.SyncBatchNorm`.
    """
    def __init__(self, *args, **kwargs):
        super(BatchNorm1d, self).__init__(*args, **kwargs)

class BatchNorm2d(SyncBatchNorm):
    r"""
    .. warning::
        BatchNorm2d is deprecated in favor of :class:`encoding.nn.SyncBatchNorm`.
    """
    def __init__(self, *args, **kwargs):
        super(BatchNorm2d, self).__init__(*args, **kwargs)

class BatchNorm3d(SyncBatchNorm):
    r"""
    .. warning::
        BatchNorm3d is deprecated in favor of :class:`encoding.nn.SyncBatchNorm`.
    """
    def __init__(self, *args, **kwargs):
        super(BatchNorm3d, self).__init__(*args, **kwargs)
