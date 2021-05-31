import torch
import numpy as np
import cupy as cp 
from functions import noso, dtdu 

def SNN_folded(cfg_fc, 
                tau_m, 
                tau_s, 
                thresh, 
                num_steps, 
                inputs, 
                batch_size, 
                w, 
                grad_eval=True):

    nlayer = len(cfg_fc)-1
    vm = [cp.zeros((batch_size, cfg_fc[i])) for i in range(nlayer)]
    vs = [cp.zeros((batch_size, cfg_fc[i])) for i in range(nlayer)]
    um = [cp.zeros((batch_size, cfg_fc[i+1])) for i in range(nlayer)]
    us = [cp.zeros((batch_size, cfg_fc[i+1])) for i in range(nlayer)]
    # membrane potential: mem = um - us
    mem = [cp.zeros((batch_size, cfg_fc[i+1])) for i in range(nlayer)]              
    # Spike record
    sp = [cp.zeros((batch_size, cfg_fc[i+1]),dtype=bool) for i in range(nlayer)]   
    # Refractory flag
    ref = [cp.ones((batch_size, cfg_fc[i+1])) for i in range(nlayer)]               
    # Cumulative number of spikes
    spike_sum = [cp.zeros((batch_size, cfg_fc[i+1])) for i in range(nlayer)]        
    # Output spike timings
    outt = num_steps * cp.ones((batch_size, cfg_fc[-1]))                              
    # Vectorize input tensor
    inputs = inputs.view(batch_size,-1)    
    # Convert input tensor to cupy array                                         
    inputs = cp.asarray(inputs)                                                     

    if grad_eval:
        # Initialization of dtdu array
        dtdu_append = [cp.zeros((batch_size, cfg_fc[i+1])) for i in range(nlayer)]
        # Initialization of v matrix to store v array
        v = [cp.zeros((batch_size, cfg_fc[i+1], cfg_fc[i])) for i in range(nlayer)]
        # Initialization of v matrix to store dvdt matrix
        dvdt_append = [cp.zeros((batch_size, cfg_fc[i+1], cfg_fc[i])) for i in range(nlayer)]

        for t in range(num_steps):
            # Poisson spike generation
            sp_in = inputs > cp.random.rand(inputs.shape[0], inputs.shape[1])
            # Calculation of first hidden layer
            ref[0], vm[0], vs[0], um[0], us[0], mem[0], sp[0] = noso(tau_m, tau_s, thresh, sp_in, ref[0], vm[0], vs[0], w[0], sp[0])
            # dtdu evaluation upon postsynaptic spiking and accumulation of dtdu onto dtdu_append
            dtdu_append[0] = dtdu(tau_m, tau_s, sp[0], um[0], us[0], dtdu_append[0])
            # v evaluation upon postsynaptic spiking and accumulation of v onto v matrix
            v[0] += cp.expand_dims((vm[0] - vs[0]),axis=1) * cp.expand_dims(sp[0].astype(float),axis=2)
            # dvdt evaluation upon postsynaptic spiking and accumulation of dvdt onto dvdt_append
            dvdt_append[0] += cp.expand_dims((vm[0]/tau_m - vs[0]/tau_s),axis=1) * cp.expand_dims(sp[0].astype(float),axis=2)
            spike_sum[0] += sp[0]
            for i in range(1, nlayer):
                ref[i], vm[i], vs[i], um[i], us[i], mem[i], sp[i] = noso(tau_m, tau_s, thresh, sp[i-1], ref[i], vm[i], vs[i], w[i], sp[i])
                dtdu_append[i] = dtdu(tau_m, tau_s, sp[i], um[i], us[i], dtdu_append[i])
                v[i] += cp.expand_dims((vm[i] - vs[i]),axis=1) * cp.expand_dims(sp[i].astype(float),axis=2)
                dvdt_append[i] += cp.expand_dims((vm[i]/tau_m - vs[i]/tau_s),axis=1) * cp.expand_dims(sp[i].astype(float),axis=2)
                spike_sum[i] += sp[i]
            # Recoding of output spike timings
            outt += sp[-1].astype(float)*(t - num_steps)
        return outt, dtdu_append, v, dvdt_append, spike_sum
    else:
        h2_out = cp.zeros((batch_size, cfg_fc[-1])) 
        for t in range(num_steps):                   
            sp_in = inputs > cp.random.rand(inputs.shape[0], inputs.shape[1])
            #first hidden layer
            ref[0], vm[0], vs[0], um[0], us[0], mem[0], sp[0] = noso(tau_m, tau_s, thresh, sp_in, ref[0], vm[0], vs[0], w[0], sp[0])
            spike_sum[0] += sp[0]
            for i in range(1, nlayer):
                ref[i], vm[i], vs[i], um[i], us[i], mem[i], sp[i] = noso(tau_m, tau_s, thresh, sp[i-1], ref[i], vm[i], vs[i], w[i], sp[i])
                spike_sum[i] += sp[i]
            outt += sp[-1].astype(float)*(t - num_steps)     
            h2_out += sp[-1].astype(float) * mem[-1]
        return outt, h2_out, spike_sum