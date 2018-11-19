#!/bin/bash/env python
import numpy as np
import random
import operator

""" Genetic Algorithm to Optimize Neural Network Weights """
def gaNN(nn,X,y,pop_size=10,no_generations=100):
    inputSize = 39
    hiddenSize = 20
    population = {}
    best_loss = 9999
    def generateFirstPopulation(pop_size=pop_size):
        for x in range(pop_size):
            population[x] = np.random.randn(inputSize, hiddenSize)
    def fitness(weight):
        nn.W1 = weight
        loss = np.mean(np.square(y - nn.forward(X)))
        return loss 
    def selectParents(population):
        global best_loss
        population_performance = {}
        for individual in population.keys():
            population_performance[individual] = fitness(population[individual])
        sorted_pop = sorted(population_performance.items(), key=operator.itemgetter(1))
        best_loss = sorted_pop[0][1]
        nn.W1 = population[sorted_pop[0][0]]
        return sorted_pop[0][0],sorted_pop[1][0],best_loss
    
    def createOffspring(parent_names):
        child = []
        parent_1 = parent_names[0]
        parent_2 = parent_names[1]
        for x in range(inputSize):
            if random.randint(0,100) > 50:
                child.append(population[parent_1][x])
            else:
                child.append(population[parent_2][x])
        mutated = mutate(child)
        return mutated 
    def nextGeneration(parents,pop_size=pop_size):
        next_generation = {}
        next_generation[0] = (population[parents[0]])
        next_generation[1] = (population[parents[1]])
        x = 2 # Keep the previous parents 
        while len(next_generation.keys()) < pop_size:
            next_generation[x] = (createOffspring(parents))
            x +=1
        return (next_generation)
    def mutate(child):
        for x in range(39):
            if random.randint(0,100) > 98:
                child[x][random.randint(0,19)] = random.uniform(-0.5,0.5)
            else:
                continue
        return child
    
    """ Generate First Population """
    generateFirstPopulation()
    firstParents = selectParents(population)
    count = 0
    while True:
        next_gen = nextGeneration(firstParents)
        population = next_gen
        firstParents = selectParents(population)
        best_loss = firstParents[2]
        count += 1
        if best_loss < 0.05:
            return nn  
        if count == no_generations:
            generateFirstPopulation()
            firstParents = selectParents(population)
            count = 0
    return nn




    
    
    
    




    





    
    
    
    


    
    
    
   
    
    

