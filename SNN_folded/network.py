import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from functions import noso
from functions import dtdu_backward
from functions import dodt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float

class fcn(nn.Module):
    def __init__(self, task, thresh, tau_m, tau_s, num_steps, frate):
        super(fcn, self).__init__()
        self.task = task
        self.thresh = thresh
        self.tau_m = tau_m
        self.tau_s = tau_s
        self.num_steps = num_steps
        self.frate = frate
        
        if self.task == 'mnist':
            self.cfg_fc = [784, 400, 10]
            
        elif self.task == 'nmnist':
            self.cfg_fc = [34*34*2, 800, 10]
            self.num_steps = 300
            
        self.fc1 = nn.Linear(self.cfg_fc[0], self.cfg_fc[1], bias=False).float()
        self.fc2 = nn.Linear(self.cfg_fc[1], self.cfg_fc[2], bias=False).float()        
        nn.init.normal_(self.fc1.weight, mean=0, std=np.sqrt(2/self.cfg_fc[0]))
        nn.init.normal_(self.fc2.weight, mean=0, std=np.sqrt(2/self.cfg_fc[1]))
        
    def forward(self, input, batch_size):
        h1_vm = h1_vs = torch.zeros(batch_size, self.cfg_fc[0], dtype=dtype, device=device)
        h1_um = h1_us = h1_spike = torch.zeros(batch_size, self.cfg_fc[1], dtype=dtype, device=device)
        h1_sav = torch.ones(batch_size, self.cfg_fc[1], dtype=dtype, device=device)
        
        h2_vm = h2_vs = torch.zeros(batch_size, self.cfg_fc[1], dtype=dtype, device=device)
        h2_um = h2_us = h2_spike = h2_u = torch.zeros(batch_size, self.cfg_fc[2], dtype=dtype, device=device)
        h2_sav = torch.ones(batch_size, self.cfg_fc[2], dtype=dtype, device=device)
        
        # output 
        out_t = self.num_steps * torch.ones(batch_size, self.cfg_fc[2], dtype=dtype, device=device)
        out_u = torch.zeros(batch_size, self.cfg_fc[2], dtype=dtype, device=device)
        sum_sp = torch.zeros(len(self.cfg_fc))
        
        # for backprop
        h1_v = torch.zeros(batch_size, self.cfg_fc[1], self.cfg_fc[0], dtype=dtype, device=device)
        h2_v = torch.zeros(batch_size, self.cfg_fc[2], self.cfg_fc[1], dtype=dtype, device=device)

        h1_dvdt = torch.zeros(batch_size, self.cfg_fc[1], self.cfg_fc[0], dtype=dtype, device=device)
        h2_dvdt = torch.zeros(batch_size, self.cfg_fc[2], self.cfg_fc[1], dtype=dtype, device=device)
        
        h1_dtdu = torch.zeros(batch_size, self.cfg_fc[1], dtype=dtype, device=device)
        h2_dtdu = torch.zeros(batch_size, self.cfg_fc[2], dtype=dtype, device=device)
        
        for step in range(self.num_steps):
            # MNIST input encoding : Poisson spike generation
            if self.task == 'mnist':
                in_spike = (input * self.frate > torch.rand(input.size(), device=device)).view(batch_size, -1).float()
            
            # N-MNIST input encoding
            elif self.task == 'nmnist':
                in_spike = input[:,:,:,:, step].view(batch_size, -1)

            # Calculation of first hidden layer
            h1_sav, h1_vm, h1_vs, h1_um, h1_us, h1_spike = noso(self.thresh, 
                                                                self.tau_m, 
                                                                self.tau_s, 
                                                                self.fc1, 
                                                                in_spike, 
                                                                h1_sav, 
                                                                h1_vm, 
                                                                h1_vs, 
                                                                h1_spike, 
                                                                outneuron=False)
            # Calculation of output layer
            h2_sav, h2_vm, h2_vs, h2_um, h2_us, h2_u, h2_spike = noso(self.thresh, 
                                                                      self.tau_m, 
                                                                      self.tau_s, 
                                                                      self.fc2, 
                                                                      h1_spike, 
                                                                      h2_sav, 
                                                                      h2_vm, 
                                                                      h2_vs, 
                                                                      h2_spike, 
                                                                      outneuron=True)
            
            # Recoding of output spike timings and output membrane potential  
            out_t += dodt.apply(h2_spike, step, self.num_steps)
            out_u += h2_spike * h2_u

            sum_sp[0] += in_spike.sum().item()
            sum_sp[1] += h1_spike.sum().item()
            sum_sp[2] += h2_spike.sum().item()
            
            # for backprop
            h1_v += torch.unsqueeze((h1_vm - h1_vs), 1) * torch.unsqueeze(h1_spike, 2)
            h2_v += torch.unsqueeze((h2_vm - h2_vs), 1) * torch.unsqueeze(h2_spike, 2)  
            
            h1_dvdt += torch.unsqueeze((h1_vm / self.tau_m - h1_vs / self.tau_s), 1) * torch.unsqueeze(h1_spike, 2)
            h2_dvdt += torch.unsqueeze((h2_vm / self.tau_m - h2_vs / self.tau_s), 1) * torch.unsqueeze(h2_spike, 2)
                        
            h1_dtdu += dtdu_backward(h1_um, h1_us, h1_spike, self.thresh, self.tau_m, self.tau_s)
            h2_dtdu += dtdu_backward(h2_um, h2_us, h2_spike, self.thresh, self.tau_m, self.tau_s)

            if out_t.max() < self.num_steps:
                return out_t, out_u, sum_sp, [h1_v, h2_v], [h1_dvdt, h2_dvdt], [h1_dtdu, h2_dtdu]
        return out_t, out_u, sum_sp, [h1_v, h2_v], [h1_dvdt, h2_dvdt], [h1_dtdu, h2_dtdu]
    
   
