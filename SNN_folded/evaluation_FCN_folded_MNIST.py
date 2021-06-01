import torchvision
import torchvision.transforms as transforms
import os
import time
import torch
import numpy as np
import cupy as cp 
from network import SNN_folded
from utils import CEloss, num_correct

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Hyperparameters
thresh = .05    # Spiking threshold
tau_m = 80      # Time constant of membrane potential kernel [ms]
tau_s = 20      # Time constant of synaptic current kernel   [ms]   
const = tau_m / (tau_m - tau_s)
decay_m, decay_s = np.exp(-1 / tau_m), np.exp(-1 / tau_s)
batch_size = 100    # Batch size
learning_rate = 1E-2    # Initial learning rate
num_epochs = 200        # Maximum number of epochs
num_steps = 32          # Number of time steps
frate = 0.2             # Maximum input firing [x1000 Hz]
reg = 1E-2              # Weight decay (L2 regularization) rate
lr_decay_rate = 0.5     # Learning rate decay rate
lr_decay_interval = 10   # Learning rate decay interval
min_lr = 5E-4           # Minimum learning rate

cfg_fc = [784, 400, 10] # FCN structure
nlayer = len(cfg_fc)-1  # Number of layers

# Loading pre-trained weights
w = [cp.load('./Pretrained_params/w1.npy'), cp.load('./Pretrained_params/w2.npy')] # Weight initialization

data_path =  './Dataset/'
test_set = torchvision.datasets.MNIST(root= data_path, train=False, download=True, transform=transforms.ToTensor())
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)

loss_test_epoch = 0 # Test loss per epoch
correct = 0
total = 0
# Test network    
for batch_idx, (inputs, targets) in enumerate(test_loader):        
    outputs = SNN_folded(cfg_fc, 
                          tau_m, 
                          tau_s, 
                          thresh, 
                          num_steps, 
                          inputs, 
                          batch_size, 
                          w, 
                          grad_eval=False)  
    # outputs[0]: output spike timings
    # outputs[1]: mebrane potential of output neurons when spiking
    # outputs[2]: spike map over SNN
    
    correct += num_correct(outputs, targets)    # Number of correct classification cases
    total += batch_size    
    loss = CEloss(outputs[0], targets, batch_size)[1]     # Gradient of loss and loss evaluations
    loss_test_epoch += loss      # Summation of losses over epochs 

print('Test Accuracy on test images: %.3f' % (100 * correct / total))
print('Test loss: %.3f' % (loss_test_epoch))
