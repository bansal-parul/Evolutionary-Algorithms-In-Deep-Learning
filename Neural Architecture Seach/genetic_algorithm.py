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
from nn_network import NeuralNetwork


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Chromosome:
    def __init__(self, params):
        self.params = params
        self.model = None
    def create_chromosome(self):
        self.model = NeuralNetwork(self.params)
          
        
def create_population(num_chromosomes, hidden_layers):
    
    population = []
    layer_names = ["Conv2d","MaxPool2d","AvgPool2d"]
    activations = ["ELU","LeakyReLU","PReLU","ReLU","RReLU","Tanh"]
    count = 0
    while(len(population) < num_chromosomes):
        count = count+1
        chromosome = []
        if count >1000:
            return population
        for i in range(hidden_layers):
            layer_params = [random.randint(1,7), random.randint(1,3),random.randint(0,3)]
            layer_params[2] = min(int(layer_params[0]/2),layer_params[2])
            activation = activations[random.randint(0,len(activations)-1)]
            layer_name_index = random.choices([0,1,2],weights = [0.5,0.3,0.2], k =1)[0]
            layer_name = layer_names[layer_name_index]
            gene = [layer_name,layer_params,activation]
            chromosome.append(gene)
        
        if chromosome not in population:
            population.append(chromosome)
    population_chromosome = []
    for params in population:
        chromosome = Chromosome(params)
        chromosome.create_chromosome()
        population_chromosome.append(chromosome)
        
    return population_chromosome


class GeneticAlgorithm:
    def __init__(self, population, local_mutation_prob, global_mutation_prob):
        self.population = population
        #self.local_mutation_prob = local_mutation_prob
        #self.global_mutation_prob =  global_mutation_prob
    

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
            
            selected_individuals.append((population[index],fitness[index]))
            current_pointer += pointer_distance
        
        return selected_individuals
    
    def crossover(self,parent1, parent2, num_offsprings, fitness,hidden_layers):
        offsprings = []
        count = 0
        while len(offsprings) <num_offsprings :
            offspring = []
            count = count+1
            if count>=1000:
                return None
            for gene_index in range(hidden_layers):
                parent = random.choices([0,1],weights = fitness , k =1)[0]
                if parent:
                    offspring.append(parent1.params[gene_index])
                else:
                    offspring.append(parent2.params[gene_index])
            if offspring not in offsprings:
                offsprings.append(offspring)
        offsprings_gene = []
        for params in offsprings:
            off_gene = Chromosome(params)
            off_gene.create_chromosome()
            offsprings_gene.append(off_gene)
        return offsprings_gene

    def evaluate_fitness(self, indiviudal,train_loader,val_loader,num_epochs=5, lr = 0.01):
        torch.cuda.manual_seed(42)
        torch.manual_seed(42)
        model = indiviudal.model
        model = model.to(device)
        optimizer_name = torch.optim.Adam
    
        optimizer = optimizer_name( model.parameters(), lr=lr)

        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(num_epochs):
            model.train()
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
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

        accuracy = correct / total
        return accuracy
    

    

