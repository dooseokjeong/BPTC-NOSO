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
frate = 0.2             # Maximum input firing rate [x1000 Hz]
reg = 1E-2              # Weight decay (L2 regularization) rate
lr_decay_rate = 0.5     # Learning rate decay rate
lr_decay_interval = 20   # Learning rate decay interval
min_lr = 1E-4           # Minimum learning rate

cfg_fc = [784, 400, 10] # FCN structure
nlayer = len(cfg_fc)-1  # Number of layers
w = [cp.random.normal(loc=0, scale=cp.sqrt(2/cfg_fc[i]), size=(cfg_fc[i+1], cfg_fc[i])) for i in range(nlayer)] # Weight initialization

def lr_scheduler(learning_rate, epoch, decay_epoch, decay_rate, minlr):
    if epoch % decay_epoch == 0 and epoch >1:
        temp = learning_rate*decay_rate
        learning_rate1 = temp*(temp > minlr) + minlr*(temp <= minlr)
    else:
        learning_rate1 = learning_rate
    return learning_rate1

names = 'spiking_model_fcn'
data_path =  './Dataset/'
train_dataset = torchvision.datasets.MNIST(root= data_path, train=True, download=True, transform=transforms.ToTensor())
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
test_set = torchvision.datasets.MNIST(root= data_path, train=False, download=True, transform=transforms.ToTensor())
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)

best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
loss_train_epoch = 0 # Training loss per epoch
loss_test_epoch = 0 # Test loss per epoch

# List initialization
acc_record = list([])
loss_train_record = list([])
loss_test_record = list([])
spike_train_record = list([])
spike_test_record = list([])

for epoch in range(num_epochs):
    running_loss = 0
    loss_train_epoch = 0
    loss_test_epoch = 0
    spike_train_epoch = 0
    spike_test_epoch = 0
    start_time = time.time()
    for i, (images, labels) in enumerate(train_loader):
        # Forward propagation       
        outputs = SNN_folded(cfg_fc, 
                              tau_m, 
                              tau_s, 
                              thresh, 
                              num_steps,
                              frate,
                              images, 
                              batch_size, 
                              w, 
                              grad_eval=True)
        # outputs[0]: output spike timings
        # outputs[1]: dtdu array
        # outputs[2]: dvdt array
        # outputs[3]: spike map over SNN    
        dldt, loss = CEloss(outputs[0], labels, batch_size) # Gradient of loss and loss evaluations
        spike_train_epoch += outputs[3][0].sum() + outputs[3][1].sum() # Summation of spikes generated over batches
        
        # Backpropagation of errors based on temporal code
        learning_rate = lr_scheduler(learning_rate, epoch, decay_epoch=lr_decay_interval, decay_rate=lr_decay_rate, minlr=min_lr)
        delta = dldt * outputs[1][-1]   # Evaluation of errors of output layer 
        dw1 = -learning_rate * outputs[2][-1] * cp.expand_dims(delta, axis=2)   # \delta_w evaluation for W(2)        
        dtemp = cp.matmul(cp.transpose(w[-1]*outputs[3][-1],(0,2,1)),cp.expand_dims(delta, axis=2)) # Backpropagation of erros toward first hidden layer
        delta = cp.squeeze(dtemp,axis=2) * outputs[1][0]    # Evaluation of errors of first hidden layer
        dw0 = -learning_rate * outputs[2][0] * cp.expand_dims(delta, axis=2)    # \delta_w evaluation for W(1)
        
        w[-1] += cp.mean(dw1,axis=0)            # w(2) update
        w[-1] -= learning_rate * reg * w[-1]    # L2 regularization
        w[0] += cp.mean(dw0,axis=0)             # w(1) update
        w[0] -= learning_rate * reg * w[0]      # L2 regularization

        running_loss += loss
        loss_train_epoch += loss
        if (i+1)%100 == 0:
             print ('Epoch [%d/%d], Iteration [%d/%d], Loss: %.5f'
                    %(epoch+1, num_epochs, i+1, len(train_dataset)//batch_size,running_loss))
             running_loss = 0  
    print('Time elapsed:', time.time()-start_time)
    correct = 0
    total = 0
    # Test network    
    for batch_idx, (inputs, targets) in enumerate(test_loader):        
        outputs = SNN_folded(cfg_fc, 
                              tau_m, 
                              tau_s, 
                              thresh, 
                              num_steps,
                              frate,
                              inputs, 
                              batch_size, 
                              w, 
                              grad_eval=False)  
        # outputs[0]: output spike timings
        # outputs[1]: mebrane potential of output neurons when spiking
        # outputs[2]: spike map over SNN
        
        correct += num_correct(outputs, targets)    # Number of correct classification cases
        total += batch_size
        spike_test_epoch += outputs[2][0].sum() + outputs[2][1].sum()   # Summation of spikes generated over batches
        dldt, loss = CEloss(outputs[0], targets, batch_size)     # Gradient of loss and loss evaluations
        loss_test_epoch += loss      # Summation of losses over epochs 

    print('\n')
    print('Epoch num:', epoch)
    print('Test Accuracy on test images: %.3f' % (100 * correct / total))
    print('Training loss: %.3f, Test loss: %.3f' % (loss_train_epoch, loss_test_epoch))
    acc = 100 * correct / total
    acc_record.append(acc)
    loss_train_record.append(loss_train_epoch)
    loss_test_record.append(loss_test_epoch)
    spike_train_record.append(spike_train_epoch)
    spike_test_record.append(spike_test_epoch)
    if (epoch+1) % 10 == 0:
        print(acc)
        print('Saving..')
        state = {
            'acc': acc,
            'epoch': epoch,
            'acc_record': acc_record,
            'best accuracy': max(acc_record),
            'loss_train_record': loss_train_record,
            'loss_test_record': loss_test_record,
            'Spike_train_record': spike_train_record,
            'Spike_test_record': spike_test_record 
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt' + names + '.t7')
        best_acc = acc
