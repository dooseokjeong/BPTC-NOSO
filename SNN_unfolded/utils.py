from __future__ import print_function
import torchvision
import torchvision.transforms as transforms
import os
import torch

from networks import fcn
from networks import cnn

from pysnn.datasets import nmnist_train_test

def load_data(task, batch_size):
    # Load MNIST dataset
    if task == 'mnist':
        
        data_path =  './Dataset/MNIST/'
        
        train_dataset = torchvision.datasets.MNIST(root= data_path, train=True, download=True, transform=transforms.ToTensor())    
        test_set = torchvision.datasets.MNIST(root= data_path, train=False, download=True, transform=transforms.ToTensor())
        
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Load N-MNIST dataset
    elif task == 'nmnist':
        
        data_path = './Dataset/NMNIST'  # nmnist data directory
        
        if os.path.isdir(data_path):
            train_dataset, test_dataset = nmnist_train_test(data_path, height=34, width=34)
        else:
            raise NotADirectoryError(
                "Make sure to download the N-MNIST dataset from https://www.garrickorchard.com/datasets/n-mnist and put it in the 'nmnist' folder."
            )
            
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, test_loader


def make_model(network, task, thresh, tau_m, tau_s, num_steps, frate):
    if network == 'fcn':
        return fcn(task, thresh, tau_m, tau_s, num_steps, frate)
    elif network == 'cnn':
        return cnn(task, thresh, tau_m, tau_s, num_steps, frate)
    else:
        print('Enter fcn or cnn')


def load_model(names, model):
    PATH =  './checkpoint/' + names
    checkpoint = torch.load(PATH)
    model.load_state_dict(checkpoint['net'])
    # epoch = checkpoint['epoch']
    # acc = checkpoint['acc']
    # acc_hist = checkpoint['acc_hist']
    # loss_train_hist = checkpoint['loss_train_hist']
    # loss_test_hist = checkpoint['loss_test_hist']
    # spike_train_hist = checkpoint['spike_train_hist']
    # spike_test_hist = checkpoint['spike_test_hist']
    
    return model


def save_model(names, model, acc, epoch, acc_hist, train_loss_hist, test_loss_hist, spike_train_hist, spike_test_hist):
    state = {
        'net': model.state_dict(),
        'acc': acc,
        'epoch': epoch,
        'acc_hist': acc_hist,
        'loss_train_hist': train_loss_hist,
        'loss_test_hist': test_loss_hist,
        'spike_train_hist': spike_train_hist,
        'spike_test_hist': spike_test_hist 
    }
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    
    torch.save(state, './checkpoint/' + names)
    
    best_acc = 0  # best test accuracy
    best_acc = max(acc_hist)
    
    if acc == best_acc:
        torch.save(state, './checkpoint/' + names + '.best')
        
        
