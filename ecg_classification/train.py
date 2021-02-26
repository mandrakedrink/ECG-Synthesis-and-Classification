import os
import time
import random

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW, Adam
from torch.optim.lr_scheduler import (CosineAnnealingLR,
                                      CosineAnnealingWarmRestarts,
                                      StepLR,
                                      ExponentialLR)

from .meter import Meter
from .dataset import ECGDataset
from .models import *
from .config import Config, seed_everything


class Trainer:
    def __init__(self, net, lr, batch_size, num_epochs):
        self.net = net.to(config.device)
        self.num_epochs = num_epochs
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = AdamW(self.net.parameters(), lr=lr)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=num_epochs, eta_min=5e-6)
        self.best_loss = float('inf')
        self.phases = ['train', 'val']
        self.dataloaders = {
            phase: get_dataloader(phase, batch_size) for phase in self.phases
        }
        self.train_df_logs = pd.DataFrame()
        self.val_df_logs = pd.DataFrame()
    
    def _train_epoch(self, phase):
        print(f"{phase} mode | time: {time.strftime('%H:%M:%S')}")
        
        self.net.train() if phase == 'train' else self.net.eval()
        meter = Meter()
        meter.init_metrics()
        
        for i, (data, target) in enumerate(self.dataloaders[phase]):
            data = data.to(config.device)
            target = target.to(config.device)
            
            output = self.net(data)
            loss = self.criterion(output, target)
                        
            if phase == 'train':
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            
            meter.update(output, target, loss.item())
        
        metrics = meter.get_metrics()
        metrics = {k:v / i for k, v in metrics.items()}
        df_logs = pd.DataFrame([metrics])
        confusion_matrix = meter.get_confusion_matrix()
        
        if phase == 'train':
            self.train_df_logs = pd.concat([self.train_df_logs, df_logs], axis=0)
        else:
            self.val_df_logs = pd.concat([self.val_df_logs, df_logs], axis=0)
        
        # show logs
        print('{}: {}, {}: {}, {}: {}, {}: {}, {}: {}'
              .format(*(x for kv in metrics.items() for x in kv))
             )
        fig, ax = plt.subplots(figsize=(5, 5))
        cm_ = ax.imshow(confusion_matrix, cmap='hot')
        ax.set_title('Confusion matrix', fontsize=15)
        ax.set_xlabel('Actual', fontsize=13)
        ax.set_ylabel('Predicted', fontsize=13)
        plt.colorbar(cm_)
        plt.show()
        
        return loss
    
    def run(self):
        for epoch in range(self.num_epochs):
            self._train_epoch(phase='train')
            with torch.no_grad():
                val_loss = self._train_epoch(phase='val')
                self.scheduler.step()
            
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                print('\nNew checkpoint\n')
                self.best_loss = val_loss
                torch.save(self.net.state_dict(), f"best_model_epoc{epoch}.pth")

              
if __name__ == '__main__':
    # init config and set random seed
    config = Config()
    seed_everything(config.seed)
     
    # init model
    #model = RNNAttentionModel(1, 64, 'lstm', False)
    #model = RNNModel(1, 64, 'lstm', True)
    model = CNN(num_classes=5, hid_size=128)  
              
    # start train
    trainer = Trainer(net=model, lr=1e-3, batch_size=96, num_epochs=30)
    trainer.run()  
              
    # write logs
    train_logs = trainer.train_df_logs
    train_logs.columns = ["train_"+ colname for colname in train_logs.columns]
    val_logs = trainer.val_df_logs
    val_logs.columns = ["val_"+ colname for colname in val_logs.columns]

    logs = pd.concat([train_logs,val_logs], axis=1)
    logs.reset_index(drop=True, inplace=True)
    logs = logs.loc[:, [
        'train_loss', 'val_loss', 
        'train_accuracy', 'val_accuracy', 
        'train_f1', 'val_f1',
        'train_precision', 'val_precision',
        'train_recall', 'val_recall']
                                     ]
    print(logs.head())
    logs.to_csv('cnn.csv', index=False)
              
