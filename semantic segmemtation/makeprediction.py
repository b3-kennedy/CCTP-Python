# -*- coding: utf-8 -*-
"""
Created on Mon Feb 21 14:54:17 2022

@author: Ben
"""

"""
Created with help from Johannes Schmidt https://towardsdatascience.com/creating-and-training-a-u-net-model-with-pytorch-for-2d-3d-semantic-segmentation-dataset-fb1f7f80fe55. 
"""


import pathlib

import numpy as np
import torch
from skimage.io import imread
from skimage.transform import resize

from prediction import predict
from transforms import normalize_01, re_normalize
import matplotlib.pyplot as plt

from training import Trainer

import segmentation_models_pytorch as smp


def get_filenames_of_path(path: pathlib.Path, ext: str = '*'):
    """Returns a list of files in a directory/path. Uses pathlib."""
    filenames = [file for file in path.glob(ext) if file.is_file()]
    return filenames


def Prediction(path, modelName, savePath):
# root directory
    root = path
    root = pathlib.Path(root)
    
    # input and target files
    images_names = get_filenames_of_path(root)
    
    
    
    images = [imread(img_name) for img_name in images_names]
    #targets = [imread(tar_name) for tar_name in targets_names]
    
    # Resize images and targets
    images_res = [resize(img, (256, 256, 3)) for img in images]
    resize_kwargs = {'order': 0, 'anti_aliasing': False, 'preserve_range': True}
    #targets_res = [resize(tar, (128, 128), **resize_kwargs) for tar in targets]
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        torch.device('cpu')
    

    aux_params=dict(
        pooling='avg',             
        dropout=0.5,               
        activation='sigmoid',      
        classes=2,                 
    )

    model = smp.Unet('resnet18', classes=2, aux_params=aux_params).to(device)
    
    
    model_name = modelName+'.pt'
    model_weights = torch.load(pathlib.Path.cwd() / model_name)
    
    model.load_state_dict(model_weights)
    

    def preprocess(img: np.ndarray):
        img = np.moveaxis(img, -1, 0)
        img = normalize_01(img)
        img = np.expand_dims(img, axis=0)
        img = img.astype(np.float32)
        return img
    
    
    def postprocess(img: torch.tensor):
        img = torch.argmax(img, dim=1)
        img = img.cpu().numpy()
        img = np.squeeze(img)
        img = re_normalize(img)
        return img
    
    output = [predict(img, model, preprocess, postprocess, device) for img in images_res]
    plt.axis('off')
    fig=plt.imshow(output[0])
    plt.savefig(savePath,bbox_inches='tight',transparent=True, pad_inches=0)



