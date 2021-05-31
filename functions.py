import torch
import numpy as np
import cupy as cp 

# dudu evaluation
def dtdu(tau_m, tau_s, sp, um, us, grad_u):
    temp = um/tau_m - us/tau_s
    temp = cp.where(temp != 0, 1/temp, temp)
    temp = cp.clip(temp, -100, 100) * sp.astype(float)
    return grad_u + temp

# NOSO model definition
def noso(tau_m, tau_s, thresh, x, ref, vm, vs, w, sp):
    const = tau_m / (tau_m - tau_s)
    decay_m, decay_s = np.exp(-1 / tau_m), np.exp(-1 / tau_s)
     
    vm = vm * decay_m + const * x.astype(float)
    vs = vs * decay_s + const * x.astype(float)
    ref = ref * (1 - sp.astype(float))
    um = cp.transpose(cp.matmul(w, cp.transpose(vm,(1,0))),(1,0))
    us = cp.transpose(cp.matmul(w, cp.transpose(vs,(1,0))),(1,0))
    mem = ref * (um - us)
    spike = mem > thresh
    return ref, vm, vs, um, us, mem, spike