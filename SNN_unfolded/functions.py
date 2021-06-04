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

    @staticmethod
    def backward(ctx, grad_output):
        u, um, us = ctx.saved_tensors
        thresh = ctx.thresh
        tau_m = ctx.tau_m 
        tau_s = ctx.tau_s
        temp = um / tau_m - us / tau_s
        temp = torch.where(temp != 0, 1 / temp, temp)
        grad_u = grad_output * u.gt(thresh).float() * torch.clamp(temp, -100, 100)
        return grad_u, None, None, None, None, None


# define the gradient of v with spike timing 
class dvdt(torch.autograd.Function):

    @staticmethod
    def forward(ctx, v, x, tau, const):
        v = v * np.exp(-1 / tau) + const * x
        ctx.tau = tau
        ctx.const = const
        return v

    @staticmethod
    def backward(ctx, grad_output):
        tau = ctx.tau
        const = ctx.const
        grad_input = grad_output.clone()
        grad_t = const * grad_input / tau
        grad_v = grad_input * np.exp(-1 / tau)
        return grad_v, grad_t, None, None
    
    
# define output time function
class dodt(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input_spike, input_time, num_steps):
        ctx.save_for_backward(input_spike)
        return input_spike * (input_time - num_steps)

    @staticmethod
    def backward(ctx, grad_output):
        input_spike, = ctx.saved_tensors
        grad_input = grad_output.clone()
        return grad_input * input_spike, None, None

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
