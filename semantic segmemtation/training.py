 # -*- coding: utf-8 -*-
"""
Created on Wed Feb 16 16:19:05 2022

@author: Ben
"""

"""
Created with help from Johannes Schmidt https://towardsdatascience.com/creating-and-training-a-u-net-model-with-pytorch-for-2d-3d-semantic-segmentation-dataset-fb1f7f80fe55. 
"""

import torch
import numpy as np
import segmentation_models_pytorch as smp
import pathlib
from skimage.transform import resize
# from transforms import normalize_01, re_normalize
from prediction import predict
from skimage.io import imread
from skimage.transform import resize
import matplotlib.pyplot as plt



class Trainer:
    def __init__(self,
                 model: torch.nn.Module,
                 device: torch.device,
                 criterion: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 training_DataLoader: torch.utils.data.Dataset,
                 validation_DataLoader: torch.utils.data.Dataset = None,
                 epochs: int = 100,
                 epoch: int = 0,
                 validation_loss_min: float = np.inf
                 ):

        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.training_DataLoader = training_DataLoader
        self.validation_DataLoader = validation_DataLoader
        self.device = device
        self.epochs = epochs
        self.epoch = epoch
        self.validation_loss_min = validation_loss_min

        self.training_loss = []
        self.validation_loss = []
        self.learning_rate = []
        

    def run_trainer(self):

        from tqdm import tqdm, trange

        progressbar = trange(self.epochs, desc='Progress')
        for i in progressbar:
            self.epoch += 1

            self._train()

            if self.validation_DataLoader is not None:
                self._validate()
        return self.training_loss, self.validation_loss, self.learning_rate

    def _train(self):


        from tqdm import tqdm, trange

        self.model.train()
        train_losses = []
        batch_iter = tqdm(enumerate(self.training_DataLoader), 'Training', total=len(self.training_DataLoader),
                          leave=False)

        for i, (x, y) in batch_iter:
            inp, target = x.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            out, x = self.model(inp)
            loss = self.criterion(out, target)
            loss_value = loss.item()
            train_losses.append(loss_value)
            loss.backward()
            self.optimizer.step()

            batch_iter.set_description(f'Training: (loss {loss_value:.4f})')

        self.training_loss.append(np.mean(train_losses))
        self.learning_rate.append(self.optimizer.param_groups[0]['lr'])
    

        batch_iter.close()




    def _validate(self):        

        from tqdm import tqdm, trange

        self.model.eval()
        valid_losses = []
        batch_iter = tqdm(enumerate(self.validation_DataLoader), 'Validation', total=len(self.validation_DataLoader),
                          leave=False)

        for i, (x, y) in batch_iter:
            inp, target = x.to(self.device), y.to(self.device)

            with torch.no_grad():
                out,x = self.model(inp)
                loss = self.criterion(out, target)
                loss_value = loss.item()
                valid_losses.append(loss_value)

                batch_iter.set_description(f'Validation: (loss {loss_value:.4f})')

        self.validation_loss.append(np.mean(valid_losses))
        
        mean_valid_loss = np.mean(valid_losses)
        print("\n loss this epoch" + str(mean_valid_loss))
        
        if  self.epoch > 1 and mean_valid_loss <= self.validation_loss_min:
            print("\n saving model..." )
            print("\n" + str(self.validation_loss_min) + " " + str(loss_value))
            model_name =  'GrassDoprout18.pt'
            torch.save(self.model.state_dict(), pathlib.Path.cwd() / model_name)
            self.validation_loss_min = mean_valid_loss     
        print("\n min val loss: " + str(self.validation_loss_min))
        batch_iter.close()
        
    def graph(self):
        from graph import plot_training
        fig = plot_training(self.training_loss, self.validation_loss, gaussian=True,sigma = 1, figsize=(10,4))
        fig.show()
        
        
    def loadmodel(self):
        model_name = 'building.pt'
        model_weights = torch.load(pathlib.Path.cwd() / model_name)
        return model_weights
        



        