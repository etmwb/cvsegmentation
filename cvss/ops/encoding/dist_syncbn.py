##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Hang Zhang
## Email: zhanghang0704@gmail.com
## Copyright (c) 2020
##
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import torch
from torch.autograd.function import Function
from . import encoding_ext

class dist_syncbatchnorm_(Function):
    @staticmethod
    def forward(ctx, x, gamma, beta, running_mean, running_var, eps, momentum, training, process_group):
        x = x.contiguous()
        ctx.training = training
        ctx.momentum = momentum
        ctx.eps = eps
        ctx.process_group = process_group

        if not ctx.training:
            _ex, _var = running_mean.contiguous(), running_var.contiguous()
            _exs = _var + _ex ** 2 
            y = encoding_ext.batchnorm_forward(x, _ex, _exs, gamma, beta, ctx.eps)
            ctx.save_for_backward(x, _ex, _exs, gamma, beta)
            return y

        size = x.numel() // x.size(1)
        if size == 1:
            raise ValueError('Expected more than 1 value per channel when training, got input size {}'.format(size))

        _ex, _exs = encoding_ext.expectation_forward(x)

        count = torch.Tensor([1]).to(x.device)
        count_all_reduce = torch.distributed.all_reduce(count, group=process_group, async_op=True)
        _ex_all_reduce = torch.distributed.all_reduce(_ex, group=process_group, async_op=True)
        _exs_all_reduce = torch.distributed.all_reduce(_exs, group=process_group, async_op=True)

        count_all_reduce.wait()
        _ex_all_reduce.wait()
        _exs_all_reduce.wait()

        _ex = _ex / count
        _exs = _exs / count

        # Update running stats
        _var = _exs - _ex ** 2
        running_mean.mul_((1 - ctx.momentum)).add_(ctx.momentum * _ex)
        running_var.mul_((1 - ctx.momentum)).add_(ctx.momentum * _var)

        # Mark in-place modified tensors
        ctx.mark_dirty(running_mean, running_var)

        # BN forward + activation
        y = encoding_ext.batchnorm_forward(x, _ex, _exs, gamma, beta, ctx.eps)

        ctx.save_for_backward(x, _ex, _exs, gamma, beta)
        return y

    @staticmethod
    def backward(ctx, dz):
        x, _ex, _exs, gamma, beta = ctx.saved_tensors
        dz = dz.contiguous()

        # BN backward
        dx, _dex, _dexs, dgamma, dbeta = \
            encoding_ext.batchnorm_backward(dz, x, _ex, _exs, gamma, beta, ctx.eps)

        if ctx.training:
            process_group = ctx.process_group
            count = torch.Tensor([1]).to(x.device)
            count_all_reduce = torch.distributed.all_reduce(count, group=process_group, async_op=True)
            _dex_all_reduce = torch.distributed.all_reduce(_dex, group=process_group, async_op=True)
            _dexs_all_reduce = torch.distributed.all_reduce(_dexs, group=process_group, async_op=True)

            count_all_reduce.wait()
            _dex_all_reduce.wait()
            _dexs_all_reduce.wait()

            _dex = _dex / count
            _dexs = _dexs / count

            dx_ = encoding_ext.expectation_backward(x, _dex, _dexs)
            dx = dx + dx_

        return dx, dgamma, dbeta, None, None, None, None, None, None

dist_syncbatchnorm = dist_syncbatchnorm_.apply
