import numpy
import torch
import os
import random
import os
import torch.nn as nn
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
import torchvision.models as models

torch.manual_seed(42)
class NeuralNetwork(nn.Module):
   
    def __init__(self, chromosome):
        super(NeuralNetwork, self).__init__()
        self.channels = 3
        self.current_shape = (225,225)
        self.layers = nn.ModuleList()
        self.chromosome = chromosome
        self.build_network()

    def get_layer_activation(self,gene):
        layer_name = gene[0]
        kernel_size, stride, padding = gene[1][0], gene[1][1], gene[1][2]
        activation = gene[2]
        in_channels = self.channels

        out_channels = min(2*in_channels,512)
        if in_channels < 10:
            out_channels = 16
        elif in_channels <=32:
            out_channels = 2*in_channels
        else:
            double_val = random.choices([0,1],weights = [0.5,0.5], k =1)[0]
            if double_val:
                out_channels = 2*in_channels
            else:
                out_channels = in_channels
        out_channels = min(512, out_channels)
        activation_layer = None
        
        current_shape = self.calculate_new_shape(self.current_shape, kernel_size, stride, padding, dilation=1)
        if current_shape[0]<3:
            return None, None,self.current_shape
        if 0 in current_shape:
            return None, None,self.current_shape
        
        
        if layer_name == "Conv2d":
            
            layer = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation=1)
            self.channels = out_channels
            activation_layer = getattr(nn, activation)()
        elif layer_name == "FractionalMaxPool2d":
            output_size = self.calculate_new_shape(self.current_shape, kernel_size, stride, padding, dilation=1)
            if 0 in output_size:
                return None ,None,self.current_shape
            layer = nn.FractionalMaxPool2d(kernel_size= kernel_size, output_size = output_size)
            
        else:
            layer_class  = getattr(nn, layer_name)
            layer = layer_class(kernel_size, stride, padding)
        
        return layer, activation_layer, current_shape
    

    def build_network(self):
        for gene in self.chromosome:
            layer,activation_layer, current_shape = self.get_layer_activation(gene)
            if layer is not None:
                self.layers.append(layer)
                self.current_shape = current_shape
            if activation_layer is not None:
                self.layers.append(activation_layer)
        #print(self.channels, self.current_shape)
        self.layers.append(nn.Flatten())
        self.layers.append(nn.Linear(self.channels * self.current_shape[0] * self.current_shape[1],5))

    def calculate_new_shape(self, shape, kernel_size, stride, padding, dilation):
        h, w = shape
        new_h = ((h + (2 * padding) - (dilation * (kernel_size - 1)) - 1) // stride) + 1
        new_w = ((w + (2 * padding) - (dilation * (kernel_size - 1)) - 1) // stride) + 1
        return new_h, new_w

    def forward(self, x):
        for layer in self.layers:
            #print(layer)
            #print(x.shape)
            x = layer(x)
            
           
        return x

