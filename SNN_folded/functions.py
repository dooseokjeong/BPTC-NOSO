import torch
import numpy as np

# define the gradient of spike timing with membrane potential
class dtdu(torch.autograd.Function):

    @staticmethod
    def forward(ctx, u, um, us, thresh, tau_m, tau_s):
        ctx.thresh = thresh
        ctx.tau_m = tau_m 
        ctx.tau_s = tau_s
        ctx.save_for_backward(u, um, us)
        return u.gt(thresh).float()

# define the gradient of v with spike timing 
class dvdt(torch.autograd.Function):

    @staticmethod
    def forward(ctx, v, x, tau, const):
        v = v * np.exp(-1 / tau) + const * x
        ctx.tau = tau
        ctx.const = const
        return v
        
# define output time function
class dodt(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input_spike, input_time, num_steps):
        ctx.save_for_backward(input_spike)
        return input_spike * (input_time - num_steps)

def dtdu_backward(um, us, sp, thresh, tau_m, tau_s):
    temp = um / tau_m - us / tau_s
    temp = torch.where(temp != 0, 1 / temp, temp)
    dtdu = sp * torch.clamp(temp, -100, 100)
    return dtdu

# Cross-entropy loss
def CEloss(outt, labels, batch_size):
    smax = torch.exp(-outt) / torch.sum(torch.exp(-outt), 1, keepdim=True)
    labels_ = torch.zeros(batch_size, 10).scatter_(1, labels.view(-1, 1), 1)
    dldt = labels_ - smax
    loss = - torch.mean(torch.sum(labels_ * torch.log(smax), 1))
    return dldt, loss

# NOSO model definition
def noso(thresh, tau_m, tau_s, ops, x, sav, vm, vs, spike, outneuron=False):
    const = tau_m / (tau_m - tau_s)
    vm = dvdt.apply(vm, x, tau_m, const)
    vs = dvdt.apply(vs, x, tau_s, const)
    sav = sav * (1. - spike)
    um = ops(vm)
    us = ops(vs)
    u = (um - us) * sav
    spike = dtdu.apply(u, um, us, thresh, tau_m, tau_s)

    if outneuron==False:
        return sav, vm, vs, um, us, spike
    return sav, vm, vs, um, us, u, spike
