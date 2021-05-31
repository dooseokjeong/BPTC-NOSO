import torch
import numpy as np
import cupy as cp 

# Cross-entropy loss
def CEloss(outt, labels, batch_size):
    smax = cp.exp(-outt) / cp.expand_dims(cp.sum(cp.exp(-outt),axis=1),axis=1)
    labels_ = cp.asarray(torch.zeros(batch_size, 10).scatter_(1, labels.view(-1, 1), 1))
    dldt = labels_ - smax
    loss = - cp.mean(cp.sum(labels_ * cp.log(smax), axis=1))
    return dldt, loss

def num_correct(outputs, targets):
    outt_t = (outputs[0] == (outputs[0].min(axis=1)[:,None])).astype(float) * outputs[1]
    comp = np.argmax(cp.asnumpy(outt_t),axis=1)
    comp1 = cp.where((cp.asarray(comp) - cp.asarray(targets)) == 0, 1, 0)
    return cp.sum(comp1)