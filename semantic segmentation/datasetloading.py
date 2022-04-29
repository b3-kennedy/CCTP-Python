# -*- coding: utf-8 -*-
"""
Created on Wed Feb 16 15:32:51 2022

@author: Ben
"""

"""
Created with help from Johannes Schmidt https://towardsdatascience.com/creating-and-training-a-u-net-model-with-pytorch-for-2d-3d-semantic-segmentation-dataset-fb1f7f80fe55. 
"""


import torch
import numpy as np
from skimage.transform import resize
from torch.utils.data import DataLoader
import albumentations
from transforms import ComposeDouble, FunctionWrapperDouble, create_dense_target, normalize_01, AlbuSeg2d
from datasets import SegmentationDataSet
from sklearn.model_selection import train_test_split
import pathlib
from training import Trainer
import segmentation_models_pytorch as smp



root = 'E:/Users/Ben/Documents/CCTP/Splitaerialimaging'
root = pathlib.Path(root)



def get_filenames_of_path(path: pathlib.Path, ext: str = '*'):
    """Returns a list of files in a directory/path. Uses pathlib."""
    filenames = [file for file in path.glob(ext) if file.is_file()]
    return filenames


# input and target files
inputs = get_filenames_of_path(root / 'grassinput')
targets = get_filenames_of_path(root / 'actualgrasstarget')

trainAug = ComposeDouble([
    FunctionWrapperDouble(resize,
                          input=True,
                          target=False,
                          output_shape=(256, 256, 3)),
    FunctionWrapperDouble(resize,
                          input=False,
                          target=True,
                          output_shape=(256, 256),
                          order=0,
                          anti_aliasing=False,
                          preserve_range=True),
    AlbuSeg2d(albumentations.HorizontalFlip(p=0.5)),
    AlbuSeg2d(albumentations.VerticalFlip(p=0.5)),
    FunctionWrapperDouble(create_dense_target, input=False, target=True),
    FunctionWrapperDouble(np.moveaxis, input=True, target=False, source=-1, destination=0),
    FunctionWrapperDouble(normalize_01)
])


# validation transformations
testAug = ComposeDouble([
    FunctionWrapperDouble(resize,
                          input=True,
                          target=False,
                          output_shape=(256, 256, 3)),
    FunctionWrapperDouble(resize,
                          input=False,
                          target=True,
                          output_shape=(256, 256),
                          order=0,
                          anti_aliasing=False,
                          preserve_range=True),
    FunctionWrapperDouble(create_dense_target, input=False, target=True),
    FunctionWrapperDouble(np.moveaxis, input=True, target=False, source=-1, destination=0),
    FunctionWrapperDouble(normalize_01)
])



# random seed
random_seed = 42

# split dataset into training set and validation set
train_size = 0.8  # 80:20 split

inputs_train, inputs_valid = train_test_split(
    inputs,
    random_state=random_seed,
    train_size=train_size,
    shuffle=True)

targets_train, targets_valid = train_test_split(
    targets,
    random_state=random_seed,
    train_size=train_size,
    shuffle=True)

# dataset training
dataset_train = SegmentationDataSet(inputs=inputs_train,
                                    targets=targets_train,
                                    use_cache= False,
                                    transform=trainAug)

# dataset validation
dataset_test = SegmentationDataSet(inputs=inputs_valid,
                                    targets=targets_valid,
                                    use_cache= False,
                                    transform=testAug)

# dataloader training
dataloader_training = DataLoader(dataset=dataset_train,
                                  batch_size=3,
                                  shuffle=True)

# dataloader validation
dataloader_validation = DataLoader(dataset=dataset_test,
                                    batch_size=3,
                                    shuffle=True)


batch = dataset_train[1]
x, y = batch

x, y = next(iter(dataloader_training))



dataiter = iter(dataloader_training)
images, labels = dataiter.next()
#plt.imshow(np.transpose(images[0].numpy(), (1, 2, 0)))

# device
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    torch.device('cpu')

#model
aux_params=dict(
    pooling='avg',             # one of 'avg', 'max'
    dropout=0.5,               # dropout ratio, default is None
    activation='sigmoid',      # activation function, default is None
    classes=2,                 # define number of output labels
)

model = smp.Unet('resnet18', classes=2, aux_params=aux_params).to(device)



# criterion
criterion = torch.nn.CrossEntropyLoss()

# optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)




# trainer
trainer = Trainer(model=model,
                  device=device,
                  criterion=criterion,
                  optimizer=optimizer,
                  training_DataLoader=dataloader_training,
                  validation_DataLoader=dataloader_validation,
                  epochs=200,
                  epoch=0,
                  validation_loss_min=np.inf)





# start training
def train():
    training_losses, validation_losses, lr_rates = trainer.run_trainer()
    trainer.graph()
    
    model_name =  'GrassDropout18End.pt'
    torch.save(model.state_dict(), pathlib.Path.cwd() / model_name)
    






