from __future__ import print_function
import time
import torch
import torch.nn as nn
import numpy as np
import argparse

from functions import CEloss
from utils import load_data
from utils import load_hyperparemeter
from utils import make_model
from utils import load_model
from utils import save_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float

parser = argparse.ArgumentParser(description='BPTC+NOSO MNIST/N-MNIST')

parser.add_argument('--task', type=str, default='nmnist', help='which task to run (mnist or nmnist)')
parser.add_argument('--network', type=str, default='fcn', help='which network to run (fcn)')
parser.add_argument('--mode', type=str, default='train', help='whether to train or test')

# Hyperparameters
parser.add_argument('--thresh', type=float, default=0.05, help='Spiking threshold [mV]')
parser.add_argument('--tau_m', type=float, default=80, help='Time constant of membrane potential kernel [ms]')
parser.add_argument('--tau_s', type=float, default=20, help='Time constant of synaptic current kernel [ms]')
parser.add_argument('--batch_size', type=int, default=200, help='Batch size')
parser.add_argument('--num_epochs', type=int, default=100, help='Maximum number of epochs')
parser.add_argument('--num_steps', type=int, default=32, help='Number of time steps')
parser.add_argument('--frate', type=float, default=0.2, help='Maximum input firing rate [x1000 Hz]')
parser.add_argument('--weight_decay', type=float, default=1E-2, help='Weight decay (L2 regularization) coefficient')
parser.add_argument('--learning_rate', type=float, default=5E-2, help='Initial learning rate')
parser.add_argument('--lr_decay_rate', type=float, default=0.5, help='Learning rate decay rate')
parser.add_argument('--lr_decay_interval', type=int, default=10, help='Learning rate decay interval')
parser.add_argument('--min_lr', type=float, default=5E-4, help='Minimum learning rate')

args = parser.parse_args()

def main():
    with torch.no_grad():
        names = args.task + '_' + args.network
        train_loader, test_loader = load_data(args.task, args.batch_size)   
        criterion = nn.CrossEntropyLoss().to(device)
        
        if args.mode == 'train':
            model = make_model(args.network, args.task, args.thresh, args.tau_m, args.tau_s, args.num_steps, args.frate).to(device)
            optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_decay_interval, gamma=args.lr_decay_rate)
            
            acc_hist = list([])
            train_loss_hist = list([])
            test_loss_hist = list([])
            spike_train_hist = list([])
            spike_test_hist = list([])
            
            for epoch in range(args.num_epochs):
                start_time = time.time()
                
                train_loss, spike_map_train, lr_ = train(model, train_loader, criterion, epoch, optimizer, scheduler)
                test_loss, spike_map_test, acc = test(model, test_loader, criterion)
                
                spike_train_hist.append(spike_map_train)
                spike_test_hist.append(spike_map_test)
                acc_hist.append(acc)
                train_loss_hist.append(train_loss)
                test_loss_hist.append(test_loss)
                
                print("Epoch: {}/{}.. ".format(epoch+1, args.num_epochs).ljust(14),
                          "Train Loss: {:.3f}.. ".format(train_loss).ljust(20),
                          "Test Loss: {:.3f}.. ".format(test_loss).ljust(19),
                          "Test Accuracy: {:.3f}".format(acc))        
                print('Time elasped: %.2f' %(time.time()-start_time), '\n')
                
                save_model(names, model, acc, epoch, acc_hist, train_loss_hist, test_loss_hist, spike_train_hist, spike_test_hist)

        elif args.mode == 'eval':
            names = names + '_saved'
            thresh, tau_m, tau_s, num_steps, frate = load_hyperparemeter(names)
            model = make_model(args.network, args.task, thresh, tau_m, tau_s, num_steps, frate).to(device)
            model = load_model(names, model)
            test_loss, spike_map_test, acc = test(model, test_loader, criterion)
            print("Test Loss: {:.3f}.. ".format(test_loss).ljust(19), "Test Accuracy: {:.3f}".format(acc))
        
def train(model, train_loader, criterion, epoch, optimizer, scheduler):
    train_loss = 0
    with torch.no_grad():
        for i, (images, labels) in enumerate(train_loader):
            spike_map_train = list([])
            outputs = model(images.to(device), args.batch_size)
            
            # outputs[0]: output spike timings
            # outputs[1]: mebrane potential of output neurons when spiking
            # outputs[2]: spike map over SNN
            # outputs[3]: v array
            # outputs[4]: dvdt array
            # outputs[5]: dtdu array
            
            spike_map_train.append(outputs[2])
            dldt, loss = CEloss(outputs[0].cpu(), labels, args.batch_size)
            train_loss += loss.item() / len(train_loader)   
                 
            # Backpropagation of errors based on temporal code
            delta = dldt.to(device) * outputs[5][1]  
            dw1 = optimizer.param_groups[0]["lr"] * outputs[3][-1] * torch.unsqueeze(delta, 2)   
            dtemp = torch.matmul((model.fc2.weight.data * outputs[4][1]).permute(0,2,1), torch.unsqueeze(delta, 2)) 
            delta = torch.squeeze(dtemp, 2) * outputs[5][0]    
            dw0 = optimizer.param_groups[0]["lr"] * outputs[3][0] * torch.unsqueeze(delta, 2)    
            
            model.fc2.weight.data -= torch.mean(dw1, 0)                                                             # w(2) update
            model.fc2.weight.data -= optimizer.param_groups[0]["lr"] * args.weight_decay * model.fc2.weight.data    # L2 regularization
            model.fc1.weight.data -= torch.mean(dw0, 0)                                                             # w(1) update
            model.fc1.weight.data -= optimizer.param_groups[0]["lr"] * args.weight_decay * model.fc1.weight.data    # L2 regularization
            
            optimizer.step()        
        
        # learning rate scheduling
        scheduler.step()
        optimizer.param_groups[0]["lr"] = np.clip(optimizer.param_groups[0]["lr"], args.min_lr, args.learning_rate)
        
    return train_loss, spike_map_train, optimizer.param_groups[0]["lr"]

def test(model, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            spike_map_test = list([])
            outputs = model(inputs.to(device), args.batch_size)
                    
            # outputs[0]: output spike timings
            # outputs[1]: mebrane potential of output neurons when spiking
            # outputs[2]: spike map over SNN
            
            dldt, loss = CEloss(outputs[0].cpu(), targets, args.batch_size)
            test_loss += loss.item() / len(test_loader)
            _, predicted = (outputs[1].cpu() * (outputs[0].cpu() == (outputs[0].cpu().min(1))[0][:, None])).max(1)
            total += float(targets.size(0))
            correct += float(predicted.eq(targets).sum().item())
            acc = 100. * float(correct) / float(total)
            spike_map_test.append(outputs[2])
    
    return test_loss, spike_map_test, acc

if __name__=='__main__':
    main()
