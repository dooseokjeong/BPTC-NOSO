import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from functions import noso
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

            if out_t.max() < self.num_steps:
                return out_t, out_u, sum_sp
        return out_t, out_u, sum_sp
    
    
class cnn(nn.Module):
    def __init__(self, task, thresh, tau_m, tau_s, num_steps, frate):
        super(cnn, self).__init__()
        self.task = task
        self.thresh = thresh
        self.tau_m = tau_m
        self.tau_s = tau_s
        self.num_steps = num_steps
        self.frate = frate
        
        if self.task == 'mnist':
            self.conv1 = nn.Conv2d(1, 12, 5, bias=False).float()
            self.conv2 = nn.Conv2d(12, 64, 5, bias=False).float()
            self.fc1 = nn.Linear(1024, 10, bias=False).float()
            
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                    torch.nn.init.normal_(m.weight, mean=0.0, std=np.sqrt(2. / n))
                
                elif isinstance(m, nn.Linear):
                    n = m.weight.size()[1]
                    torch.nn.init.normal_(m.weight, mean=0.0, std=np.sqrt(2. / n))
                    
        elif self.task == 'nmnist':
            self.num_steps = 300
            self.conv1 = nn.Conv2d(2, 12, 5, bias=False).float()
            self.conv2 = nn.Conv2d(12, 64, 5, bias=False).float()
            self.fc1 = nn.Linear(1600, 10, bias=False).float()       
            
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    torch.nn.init.xavier_normal_(m.weight, gain=1.0)
    
                elif isinstance(m, nn.Linear):
                    torch.nn.init.xavier_normal_(m.weight, gain=1.0)

    def forward(self, input, batch_size):
        if self.task == 'mnist':
            hc1_vm = hc1_vs = torch.zeros(batch_size, 1, 28, 28, dtype=dtype, device=device)
            hc1_um = hc1_us = hc1_spike = torch.zeros(batch_size, 12, 24, 24, dtype=dtype, device=device)
            hc1_sav = torch.ones(batch_size, 12, 24, 24, dtype=dtype, device=device)
            
            hc2_vm = hc2_vs = torch.zeros(batch_size, 12, 12, 12, dtype=dtype, device=device)
            hc2_um = hc2_us = hc2_spike = torch.zeros(batch_size, 64, 8, 8, dtype=dtype, device=device)
            hc2_sav = torch.ones(batch_size, 64, 8, 8, dtype=dtype, device=device)
            
            hf1_vm = hf1_vs = torch.zeros(batch_size, 1024, dtype=dtype, device=device)
            hf1_um = hf1_us = hf1_u = hf1_spike = torch.zeros(batch_size, 10, dtype=dtype, device=device)
            hf1_sav = torch.ones(batch_size, 10, dtype=dtype, device=device)
            
        elif self.task == 'nmnist':
            hc1_vm = hc1_vs = torch.zeros(batch_size, 1, 32, 32, dtype=dtype, device=device)
            hc1_um = hc1_us = hc1_spike = torch.zeros(batch_size, 12, 28, 28, dtype=dtype, device=device)
            hc1_sav = torch.ones(batch_size, 12, 28, 28, dtype=dtype, device=device)
            
            hc2_vm = hc2_vs = torch.zeros(batch_size, 12, 14, 14, dtype=dtype, device=device)
            hc2_um = hc2_us = hc2_spike = torch.zeros(batch_size, 64, 10, 10, dtype=dtype, device=device)
            hc2_sav = torch.ones(batch_size, 64, 10, 10, dtype=dtype, device=device)
            
            hf1_vm = hf1_vs = torch.zeros(batch_size, 1600, dtype=dtype, device=device)
            hf1_um = hf1_us = hf1_u = hf1_spike = torch.zeros(batch_size, 10, dtype=dtype, device=device)
            hf1_sav = torch.ones(batch_size, 10, dtype=dtype, device=device)
      
        out_t = self.num_steps * torch.ones(batch_size, 10, dtype=dtype, device=device)
        out_u = torch.zeros(batch_size, 10, dtype=dtype, device=device)
        sum_sp = torch.zeros(4) 
        
        for step in range(self.num_steps): # simulation time steps            
            # MNIST input encoding : Poisson spike generation
            if self.task == 'mnist':
                in_spike = (input * self.frate > torch.rand(input.size(), device=device)).float()
            
            # N-MNIST input encoding
            elif self.task == 'nmnist':
                in_spike = input[:, :, 1:33, 1:33, step]
            
            # Calculation of first convolutional layer
            hc1_sav, hc1_vm, hc1_vs, hc1_um, hc1_us, hc1_spike = noso(self.thresh, 
                                                                      self.tau_m, 
                                                                      self.tau_s, 
                                                                      self.conv1, 
                                                                      in_spike, 
                                                                      hc1_sav, 
                                                                      hc1_vm, 
                                                                      hc1_vs, 
                                                                      hc1_spike, 
                                                                      outneuron=False)
            
            # Calculation of second convolutional layer
            hc2_sav, hc2_vm, hc2_vs, hc2_um, hc2_us, hc2_spike = noso(self.thresh, 
                                                                      self.tau_m, 
                                                                      self.tau_s, 
                                                                      self.conv2, 
                                                                      F.max_pool2d(hc1_spike, kernel_size=2), 
                                                                      hc2_sav, 
                                                                      hc2_vm, 
                                                                      hc2_vs, 
                                                                      hc2_spike, 
                                                                      outneuron=False)
            # Calculation of linear layer
            hf1_sav, hf1_vm, hf1_vs, hf1_um, hf1_us, hf1_u, hf1_spike = noso(self.thresh, 
                                                                      self.tau_m, 
                                                                      self.tau_s, 
                                                                      self.fc1, 
                                                                      F.max_pool2d(hc2_spike, kernel_size=2).view(batch_size, -1), 
                                                                      hf1_sav, 
                                                                      hf1_vm, 
                                                                      hf1_vs, 
                                                                      hf1_spike, 
                                                                      outneuron=True)

            out_t += dodt.apply(hf1_spike, step, self.num_steps)
            out_u += hf1_spike * hf1_u
            
            sum_sp[0] += in_spike.sum().item()
            sum_sp[1] += hc1_spike.sum().item()
            sum_sp[2] += hc2_spike.sum().item()
            sum_sp[3] += hf1_spike.sum().item()
            
            if out_t.max() < self.num_steps:
                return out_t, out_u, sum_sp
        return out_t, out_u, sum_sp
    
    
    
