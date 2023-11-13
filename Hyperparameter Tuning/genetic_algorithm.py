import numpy as np
import random
import torch
import torch.nn as nn
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
import copy
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class genetic_algorithm:
    def __init__(self, population, local_mutation_prob, global_mutation_prob):
        self.population = population
        self.local_mutation_prob = local_mutation_prob
        self.global_mutation_prob =  global_mutation_prob
    
    def get_mutated_val(self, gene_name, gene):
        value = gene.value
        if gene.mutation:
                value = abs(np.random.normal(gene.value, gene.mutation_variance))
        gene.value = value
        return gene
    
    def mutation(self, chromosome):
        #using uniform mutation
        for gene in list(chromosome.keys()):
            mut_ind = random.choices([0,1],weights = [self.local_mutation_prob, 1-self.local_mutation_prob], k =1)[0]
            if mut_ind:
                chromosome[gene] = self.get_mutated_val(gene,chromosome[gene])
        return chromosome
    

    def crossover(self,parent1, parent2, num_offsprings):
        offsprings = []
        while len(offsprings) <=num_offsprings :
            offspring = {}
            for gene in list(parent1.keys()):
                parent = random.choices([0,1],weights = [0.5,0.5], k =1)[0]
                if parent:
                    offspring[gene] = parent1[gene]
                else:
                    offspring[gene] = parent2[gene]
            if offspring not in offsprings:
                offsprings.append(offspring)
        return offsprings
    
    def evaluate_fitness(self, indiviudal,train_loader,val_loader,model):
        optimizer_name = indiviudal["optimiser"].value
        batch_size = int(indiviudal["batch_size"].value)
        lr = indiviudal["lr"].value
        num_epochs = int(indiviudal["num_epochs"].value)
       
        optimizer_class = getattr(optim, optimizer_name)
        optimizer = optimizer_class(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(num_epochs):
            model.train()
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                #loss = criterion(outputs[0], labels)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100*correct/total
        return accuracy


    def parent_selection(self,population, fitness, num_selections):
        total_fitness = sum(fitness)
        pointer_distance = total_fitness / num_selections
        selected_individuals = []
        start_point = np.random.uniform(0, pointer_distance)
        current_pointer = start_point
        index = 0
        for _ in range(num_selections):
            while fitness[index] < current_pointer:
                current_pointer -= fitness[index]
                index = (index + 1) % len(population)
            
            selected_individuals.append(population[index])
            current_pointer += pointer_distance
        
        return selected_individuals


        



        


            



        
        


