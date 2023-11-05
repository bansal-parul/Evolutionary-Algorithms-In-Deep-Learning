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
from genetic_algorithm import *
from nn_network import NeuralNetwork
from genetic_algorithm import *
import json

def nas():
    population_size = 100
    population = create_population(population_size,30)
    with open("./population.json", 'w') as f:
        f.write('[')
        for ind, chromosome in enumerate(population):
            json_object = { "parameters":chromosome.params}
            if ind != 0:
                f.write(',')
            json.dump(json_object, f)
        f.write(']')
    preprocess = transforms.Compose([
        transforms.Resize(225),
        transforms.CenterCrop(225),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    train_dataset = datasets.ImageFolder("/Users/parul/Documents/Parul/EA Project/Dogs/data_subset/train", transform=preprocess)
    val_dataset = datasets.ImageFolder("/Users/parul/Documents/Parul/EA Project/Dogs/data_subset/val", transform=preprocess)
    #train_dataset = datasets.ImageFolder("./Dogs/data/train", transform=preprocess)
    #val_dataset = datasets.ImageFolder("./Dogs/data/val", transform=preprocess)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last = True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True, drop_last = True)
    
    
    for i in range(80):
       
        population_size = len(population)
        print(f"=======================iteration num {i}================================")
        if population_size<=10:
            break
        ga = GeneticAlgorithm(population,0,0)
        #step 1 : Evaluate fitness
        fitness = []
        for chromosome in population:
            if i < 50:
                num_epochs = 5
            if i >=50:
                num_epochs = 20
            fitness_chrom = ga.evaluate_fitness(chromosome,train_loader,val_loader, num_epochs)
            fitness.append(fitness_chrom)
            #print(fitness)
        # for ind,chromosome in enumerate(population):
        #     print(chromosome.model)
        with open("./genetic_current_population_fitness.json", 'w') as f:
            f.write('[')
            for ind, chromosome in enumerate(population):
                json_object = { "parameters":str(chromosome.model),"fitness": fitness[ind]}
                if ind != 0:
                    f.write(',')
                json.dump(json_object, f)
            f.write(']')
        
        
        #Step 2 : create offsprings
        offsprings = []
        count = 0
        while len(offsprings) < int(0.6*population_size):
            if population_size<=1:
                break
            count = count+1
            #step 1 : select parents for mutation
            parents = ga.parent_selection(population = population, fitness = fitness, num_selections = 2)
            #step 2 : create offsprings
            offspring = ga.crossover(parents[0][0], parents[1][0], 1, [parents[0][1], parents[1][1]],hidden_layers=20)

            if offspring == None:
                continue
            if offspring not in offsprings:
                offsprings.append(offspring[0])
            if count >=10000:
                population_size = int(0.8*population_size)
                break
        
        
        #Create new population
        #sort fitness values
        indexed_list = list(enumerate(fitness))

        # Sorting the list of tuples based on the values in descending order
        sorted_indexed_list = sorted(indexed_list, key=lambda x: x[1], reverse=True)
        percent_old = int(0.4*population_size)
        none_list = [offsprings.append(population[ind[0]]) for ind in sorted_indexed_list[0:percent_old]]
        print(f"=======================fittest individuals {i}================================")
        none_list = [print(population[ind[0]].model, fitness[ind[0]]) for ind in sorted_indexed_list[0:min(5,percent_old)]]
        population = list(offsprings)
        with open(f"./population_{i+1}.json", 'w') as f:
            f.write('[')
            for ind, chromosome in enumerate(population):
                json_object = { "parameters":chromosome.params}
                if ind != 0:
                    f.write(',')
                json.dump(json_object, f)
            f.write(']')


if __name__ == '__main__':
    nas()
    
    
    
    
    
