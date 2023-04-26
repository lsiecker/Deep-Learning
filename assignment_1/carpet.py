from PIL import Image, ImageDraw, ImageFilter
import numpy as np
import matplotlib.pyplot as plt
import torch

def show_carpet(X, idx):
    '''
    Function to visualize what a carpet looks like
    
    Parameters
    ----------
        X (tensor): dataset which contains the carpet
        idx (int): index of the carpet you want to visualize
        
    '''
    carpet = X[idx,0].numpy()
    
    im1 = (np.tile(carpet, (5,5))*255).astype(np.uint8)
    im1 = np.repeat(im1, 20, axis=1)
    im1 = np.repeat(im1, 20, axis=0)
    pim1 = Image.fromarray(im1)
    
    pim2 = Image.fromarray(np.ones((3000,3000))*255)
    
    pim2.paste(pim1.rotate(45), (-2000, -2000))
    
    edge = np.ones((1000, 2000))
    edge[30:-30,30:-30] = 0
    edge[60:-60,60:-60] = 1

    im3 = Image.fromarray(edge*255)
    
    mask = np.ones_like(edge)*255
    mask[90:-90,90:-90] = 0
    msk = Image.fromarray(mask).convert("1")
    
    pim2.paste(im3, mask=msk)
    
    plt.imshow(np.array(pim2), cmap='brg_r')
    plt.xlim(0,2000)
    plt.ylim(0,1000)
    plt.axis('off')
    
    
def oh_to_label(y):
    '''
    Function to go from 1 hot encoding to class label for task 1
    
    Parameters
    ----------
        y (tensor): One hot encoded tensor of shape (n,3)
        
    Returns   
    ----------
        labels (list): list of strings
    '''
    
    cls_dict = {0: 'Convolushahr', 1:'Transformabad', 2:'Reinforciya'}
    classes = y.argmax(1).numpy().tolist()
    
    labels = [cls_dict[i] for i in classes]
    
    return labels