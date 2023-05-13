# -*- coding: utf-8 -*-
"""
Created on Fri Jul 06 01:05:05 2018

@author: bpark1
"""

import copy
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim


from methods.torch.evaluation import getScores, getPredsLabels
from time import sleep
from torchcontrib.optim import SWA
from tqdm import tqdm

class EarlyStopping():
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs.
    """
    def __init__(self, patience=5, min_delta=0):
        """
        :param patience: how many epochs to wait before stopping when loss is
               not improving
        :param min_delta: minimum difference between new loss and old loss for
               new loss to be considered as an improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    def __call__(self, val_loss):
        if self.best_loss == None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            #print(f"INFO: Early stopping counter {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                #print('INFO: Early stopping')
                self.early_stop = True
                
def validate(model, test_dataloader, epoch, args, tqdm_=False):
    model.eval()
    val_running_loss = 0.0
    val_running_acc = 0.0
    
    counter = 0
    with torch.no_grad():
        if tqdm_:
            with tqdm(test_dataloader, unit="batch") as tepoch:
                tepoch.set_description(f"Val Epoch {epoch}")

                for i, data in enumerate(tepoch):
                    counter += 1
                    inputs, labels = data
                    labels = labels.cuda()
                    inputs = inputs.cuda()

                    outputs = model(inputs)
                    loss = F.nll_loss(outputs, labels)

                    predictions = outputs.argmax(dim=1, keepdim=True).squeeze()
                    correct = (predictions == labels).sum().item()
                    accuracy = correct / args['batchSize']
                    
                    val_running_loss+=loss.item()
                    
                    val_running_acc+=100. * accuracy

                    tepoch.set_postfix(loss=loss.item(), accuracy=val_running_acc/counter, average_loss=val_running_loss/counter)
                    sleep(0.1)
        else:
            for i, data in enumerate(test_dataloader):
                counter += 1
                inputs, labels = data
                labels = labels.cuda()
                inputs = inputs.cuda()

                outputs = model(inputs)
                loss = F.nll_loss(outputs, labels)
                
                val_running_loss+=loss.item()
                
        val_loss = val_running_loss / counter
        return val_loss

def runModel_ea(net, trainLoader, valLoader, args, printBool=False, cuda=True, tqdm_=False, patience=5):
    net = net.cuda()
    optimizer = optim.Adam(net.parameters(), lr = args['lr'])
    net = net.train()
    early_stopping = EarlyStopping(patience=patience)
    
    for epoch in range(args['epochs']): 
        if tqdm_:
            with tqdm(trainLoader, unit="batch") as tepoch:
                tepoch.set_description(f"Epoch {epoch}")

                for i, data in enumerate(tepoch):
                    inputs, labels = data
                    labels = labels.cuda()
                    inputs = inputs.cuda()
                    optimizer.zero_grad()
                    outputs = net(inputs)
                    loss = F.nll_loss(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    predictions = outputs.argmax(dim=1, keepdim=True).squeeze()
                    correct = (predictions == labels).sum().item()
                    accuracy = correct / args['batchSize']

                    tepoch.set_postfix(loss=loss.item(), accuracy=100. * accuracy)
                    sleep(0.1)
        else:
            for i, data in enumerate(trainLoader):
                inputs, labels = data
                labels = labels.cuda()
                inputs = inputs.cuda()
                optimizer.zero_grad()
                outputs = net(inputs)
                loss = F.nll_loss(outputs, labels)
                loss.backward()
                optimizer.step()

        val_epoch_loss = validate(net, valLoader, epoch, args, tqdm_=tqdm_)
        
        early_stopping(val_epoch_loss)
        if early_stopping.early_stop:
            break
    return net
        
    
def runModel(net, trainLoader, valLoader, args, printBool=False, cuda=True):
    net = net.cuda()
    optimizer = optim.Adam(net.parameters(), lr = args['lr'])
    net = net.train()
    
    best_score = 0
    
    
    for epoch in range(args['epochs']): 
        
        for i, data in enumerate(trainLoader):
            inputs, labels = data
            labels = labels.cuda()
            inputs = inputs.cuda()
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = F.nll_loss(outputs, labels)
            loss.backward()
            optimizer.step()

        score = getScores(net, valLoader, probs=False, cuda=True)['f1_micro']
        
        if score > best_score:
            state_dict = net.state_dict().copy()
            best_score = score
            best_epoch = epoch
    print(f"Best epoch {best_epoch}")
            
    net.load_state_dict(state_dict)
    return net