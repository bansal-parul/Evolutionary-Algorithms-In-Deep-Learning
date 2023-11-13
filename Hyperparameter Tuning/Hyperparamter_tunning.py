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
import timm
from genetic_algorithm import *


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class gene:
    def __init__(self, value, mutation, mutation_variance):
        self.value = value
        self.mutation = mutation
        self.mutation_variance = mutation_variance
if __name__ == '__main__':
    
    population = [{"lr": gene(0.01, True, 0.01), "optimiser" : gene("Adam", False, 0),"batch_size" : gene(16, True, 8),"num_epochs" : gene(4, True, 2)},
                      {"lr": gene(0.0001, True, 0.01), "optimiser" : gene("Adam", False, 0),"batch_size" : gene(64, True, 8),"num_epochs" : gene(10, True, 2)},
                      {"lr": gene(0.1, True, 0.01), "optimiser" : gene("SGD", False, 0),"batch_size" : gene(8, True, 8),"num_epochs" : gene(8, True, 2)},
                      {"lr": gene(0.001, True, 0.01), "optimiser" : gene("SGD", False, 0),"batch_size" : gene(32, True, 8),"num_epochs" : gene(14, True, 2)},
                      {"lr": gene(0.001, True, 0.01), "optimiser" : gene("Adadelta", False, 0),"batch_size" : gene(32, True, 8),"num_epochs" : gene(12, True, 2)},
                      {"lr": gene(0.001, True, 0.01), "optimiser" : gene("Adadelta", False, 0),"batch_size" : gene(32, True, 8),"num_epochs" : gene(12, True, 2)},
                      {"lr": gene(0.0001, True, 0.01), "optimiser" : gene("RMSprop", False, 0),"batch_size" : gene(64, True, 8),"num_epochs" : gene(16, True, 2)},
                      {"lr": gene(0.1, True, 0.01), "optimiser" : gene("RMSprop", False, 0),"batch_size" : gene(8, True, 8),"num_epochs" : gene(8, True, 2)},
                      {"lr": gene(0.01, True, 0.01), "optimiser" : gene("Adam", False, 0),"batch_size" : gene(32, True, 8),"num_epochs" : gene(12, True, 2)},
                      {"lr": gene(0.0001, True, 0.01), "optimiser" : gene("SGD", False, 0),"batch_size" : gene(128, True, 8),"num_epochs" : gene(4, True, 2)},
                      ]
    # population = [{"lr": gene(0.0001, True, 0.01), "optimiser" : gene("RMSprop", False, 0),"batch_size" : gene(61, True, 8),"num_epochs" : gene(14, True, 2)},
    #                   {"lr": gene(0.0001, True, 0.01), "optimiser" : gene("RMSprop", False, 0),"batch_size" : gene(33, True, 8),"num_epochs" : gene(12, True, 2)},
    #                   {"lr": gene(0.01, True, 0.01), "optimiser" : gene("Adadelta", False, 0),"batch_size" : gene(33, True, 8),"num_epochs" : gene(14, True, 2)},
    #                   {"lr": gene(0.01, True, 0.01), "optimiser" : gene("RMSprop", False, 0),"batch_size" : gene(33, True, 8),"num_epochs" : gene(12, True, 2)},
    #                   {"lr": gene(0.0001, True, 0.01), "optimiser" : gene("Adam", False, 0),"batch_size" : gene(61, True, 8),"num_epochs" : gene(12, True, 2)},
    #                   {"lr": gene(0.0001, True, 0.01), "optimiser" : gene("SGD", False, 0),"batch_size" : gene(37, True, 8),"num_epochs" : gene(14, True, 2)},
    #                   {"lr": gene(0.0001, True, 0.01), "optimiser" : gene("Adadelta", False, 0),"batch_size" : gene(67, True, 8),"num_epochs" : gene(14, True, 2)},
    #                   {"lr": gene(0.0001, True, 0.01), "optimiser" : gene("Adam", False, 0),"batch_size" : gene(61, True, 8),"num_epochs" : gene(14, True, 2)},
    #                   {"lr": gene(0.01, True, 0.01), "optimiser" : gene("Adadelta", False, 0),"batch_size" : gene(33, True, 8),"num_epochs" : gene(14, True, 2)},
    #                   {"lr": gene(0.0001, True, 0.01), "optimiser" : gene("Adadelta", False, 0),"batch_size" : gene(37, True, 8),"num_epochs" : gene(14, True, 2)},
    #                   ]
    preprocess = transforms.Compose([
        transforms.Resize(299),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    train_dataset = datasets.ImageFolder("./Birds/data_subset/train", transform=preprocess)
    val_dataset = datasets.ImageFolder("./Birds/data_subset/val", transform=preprocess)
    
    local_mutation_prob = 0.5
    global_mutation_prob = 0.1    
    num_interations = 20
    population_size = 10
    elite_percentage = 0.2
    
    
    model = timm.create_model('efficientnet_b3', pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    for name, param in model.named_parameters():
        if "blocks.6" in name or "blocks.7" in name or "classifier" in name:
            param.requires_grad = True
    #model.fc = nn.Linear(2048, 50)
    model.classifier = torch.nn.Linear(model.classifier.in_features, 50)

    #model = model.to(device)


    for i in range(30):
        if population_size<=1:
            break
        print(f"=======================iteration num {i}================================")
        ga = genetic_algorithm(population, local_mutation_prob,global_mutation_prob)
        fitness = []
        for chromosome in population:
            #step 1 : evaluate parents
            batch_size = int(chromosome["batch_size"].value)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
            model_chrom = model
            model_chrom = model.to(device)
            accuracy_chrom = ga.evaluate_fitness(indiviudal=chromosome,train_loader=train_loader,val_loader=val_loader,model=model_chrom)
            fitness.append(accuracy_chrom)
            print(accuracy_chrom)
        print("fitness", fitness)
        print("============================Current Population =====================================")
        for i,chromosome in enumerate(population):
            print(f"opt :{chromosome['optimiser'].value} lr: {chromosome['lr'].value} batch_size: {chromosome['batch_size'].value} num_epochs:{chromosome['num_epochs'].value} fitness: {fitness[i]} ")

        offsprings = []
        count = 0
        while len(offsprings) < int(0.8*population_size):
            if population_size<=1:
                break
            count = count+1
            #step 2 : select parents for mutation
            parents = ga.parent_selection(population = population, fitness = fitness, num_selections = 2)

            #step 3 : create offsprings
            offspring = ga.crossover(parents[0], parents[1], 1)[0]


            #step 4 : mutate offsprings
            mut_ind = random.choices([0,1],weights = [1-global_mutation_prob, global_mutation_prob], k =1)[0]
            if mut_ind:
                offspring = ga.mutation(offspring)
            if offspring not in offsprings:
                offsprings.append(offspring)
            if count >=100:
                population_size = int(0.9*population_size)

        #step 5 : create new population

        #sort fitness values
        indexed_list = list(enumerate(fitness))

        # Sorting the list of tuples based on the values in descending order
        sorted_indexed_list = sorted(indexed_list, key=lambda x: x[1], reverse=True)

        # Selecting the indices of the top two maximum values
        offsprings.append(population[sorted_indexed_list[0][0]])
        offsprings.append(population[sorted_indexed_list[1][0]])
        population = list(offsprings)
        print("============================New Population =====================================")
        for i,chromosome in enumerate(population):
            print(f"opt :{chromosome['optimiser'].value} lr: {chromosome['lr'].value} batch_size: {chromosome['batch_size'].value} num_epochs:{chromosome['num_epochs'].value} ")

        
        
        

