#!/bin/bash/env python
import random 
import operator 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt

progress = []
main_pop = None
main_pop_temp = None
#Author Cited: https://towardsdatascience.com/evolution-of-a-salesman-a-complete-genetic-algorithm-tutorial-for-python-6fe5d2b3ca35
""" Travelling sales man Genetic Algorithm """
class City:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def distance(self, city):
        xDis = abs(self.x - city.x)
        yDis = abs(self.y - city.y)
        distance = np.sqrt((xDis ** 2) + (yDis ** 2))
        return distance
    
    def __repr__(self):
        return "(" + str(self.x) + "," + str(self.y) + ")"

class Fitness:
    def __init__(self, route):
        self.route = route
        self.distance = 0
        self.fitness= 0.0
    
    def routeDistance(self):
        if self.distance ==0:
            pathDistance = 0
            for i in range(0, len(self.route)):
                fromCity = self.route[i]
                toCity = None
                if i + 1 < len(self.route):
                    toCity = self.route[i + 1]
                else:
                    toCity = self.route[0]
                pathDistance += fromCity.distance(toCity)
            self.distance = pathDistance
        return self.distance
    
    def routeFitness(self):
        if self.fitness == 0:
            self.fitness = 1 / float(self.routeDistance())
        return self.fitness
    
def createRoute(cityList):
    route = random.sample(cityList, len(cityList))
  
    return route



def initialPopulation(popSize, cityList):
    population = []
    for i in range(0, popSize):
        population.append(createRoute(cityList))
    return population

def rankRoutes(population):
    fitnessResults = {}
    for i in range(0,len(population)):
        fitnessResults[i] = Fitness(population[i]).routeFitness()
    return sorted(fitnessResults.items(), key = operator.itemgetter(1), reverse = True) 

def selection(popRanked, eliteSize):
    selectionResults = []
    df = pd.DataFrame(np.array(popRanked), columns=["Index","Fitness"])
    df['cum_sum'] = df.Fitness.cumsum()
    df['cum_perc'] = 100*df.cum_sum/df.Fitness.sum()
    
    for i in range(0, eliteSize):
        selectionResults.append(popRanked[i][0])
    for i in range(0, len(popRanked) - eliteSize):
        pick = 100*random.random()
        for i in range(0, len(popRanked)):
            if pick <= df.iat[i,3]:
                selectionResults.append(popRanked[i][0])
                break
    return selectionResults 


def matingPool(population, selectionResults):
    matingpool = []
    for i in range(0, len(selectionResults)):
        index = selectionResults[i]
        matingpool.append(population[index])
    return matingpool 

def breed(parent1, parent2):
    child = []
    childP1 = []
    childP2 = []
    
    geneA = int(random.random() * len(parent1))
    geneB = int(random.random() * len(parent1))
    
    startGene = min(geneA, geneB)
    endGene = max(geneA, geneB)

    for i in range(startGene, endGene):
        childP1.append(parent1[i])
        
    childP2 = [item for item in parent2 if item not in childP1]

    child = childP1 + childP2
    return child 


def breedPopulation(matingpool, eliteSize):
    children = []
    length = len(matingpool) - eliteSize
    pool = random.sample(matingpool, len(matingpool))

    for i in range(0,eliteSize):
        children.append(matingpool[i])
    
    for i in range(0, length):
        child = breed(pool[i], pool[len(matingpool)-i-1])
        children.append(child)
    return children 

def mutate(individual, mutationRate):
    for swapped in range(len(individual)):
        if(random.random() < mutationRate):
            swapWith = int(random.random() * len(individual))
            
            city1 = individual[swapped]
            city2 = individual[swapWith]
            
            individual[swapped] = city2
            individual[swapWith] = city1
    return individual 

def mutatePopulation(population, mutationRate):
    mutatedPop = []
    
    for ind in range(0, len(population)):
        mutatedInd = mutate(population[ind], mutationRate)
        mutatedPop.append(mutatedInd)
    return mutatedPop 


def nextGeneration(currentGen, eliteSize, mutationRate):
    popRanked = rankRoutes(currentGen)
    selectionResults = selection(popRanked, eliteSize)
    matingpool = matingPool(currentGen, selectionResults)
    children = breedPopulation(matingpool, eliteSize)
    nextGeneration = mutatePopulation(children, mutationRate)
    return nextGeneration 

def geneticAlgorithm(population, popSize, eliteSize, mutationRate, generations):
    # pop = initialPopulation(popSize, population)
    
    pop = main_pop_temp
    print("Initial distance: " + str(1 / rankRoutes(pop)[0][1]))
    print 'Optimizing Please Wait....'
    for i in range(0, generations):
        pop = nextGeneration(pop, eliteSize, mutationRate)
    
    print("Final distance: " + str(1 / rankRoutes(pop)[0][1]))
    bestRouteIndex = rankRoutes(pop)[0][0]
    bestRoute = pop[bestRouteIndex]
    return bestRoute 


""" Random Hill Search for TSP """
def randomHillSearch(population,popSize,temp_dist=9999):
    temp_dist = 99999
    progress = []
    pop = main_pop
    print("Initial distance: " + str(1 / rankRoutes(pop)[0][1]))
    print 'Optimizing Please Wait....'
    for x in range(500):
        pop = initialPopulation(popSize, population)
        if (1 / rankRoutes(pop)[0][1]) < temp_dist:
            temp_dist = (1 / rankRoutes(pop)[0][1])
            progress.append(temp_dist)
        new_route = createRoute(population)
    print "Final Distance: " , temp_dist 
    return progress


""" Simulated Annealing """
def simulatedAnnealing(population,popSize,temp_dist=9999):
    progress = []
    initialRoutes = main_pop_temp
    temp_dist = 99999
    temp_inital = initialRoutes
    def swapRoutes():
        rand_1 = random.randint(0,99)
        rand_2 = random.randint(0,24)
        rand_3 = random.randint(0,24)
        if rand_2 == rand_3:
            swapRoutes()
        temp = temp_inital[rand_1][rand_2]
        temp_inital[rand_1][rand_2] = temp_inital[rand_1][rand_3]
        temp_inital[rand_1][rand_3] = temp
    alpha = 0.7
    Temperature = 5.0
    Temperature_min = 0.00001
    
    k = 100
    print("Initial distance: " + str(1 / rankRoutes(initialRoutes)[0][1]))
    print 'Optimizing Please Wait....'
    while Temperature > Temperature_min:
        i = 0
        while i <= k:
            swapRoutes()
            new_dist = (1 / rankRoutes(temp_inital)[0][1])
            if new_dist < temp_dist:
                temp_dist = new_dist
                progress.append(temp_dist)
                temp_distances = initialRoutes
                
            else:
                if random.randint(0,20) > 100:
                    temp_inital = initialRoutes
            
            i += 1
        Temperature = Temperature * alpha
    print "Final Distance: " , temp_dist 
    return progress


def geneticAlgorithmPlot(population, popSize, eliteSize, mutationRate, generations):
    pop = main_pop_temp
    progress = []
    progress.append(1 / rankRoutes(pop)[0][1])
    
    for i in range(0, generations):
        pop = nextGeneration(pop, eliteSize, mutationRate)
        progress.append(1 / rankRoutes(pop)[0][1])
    plt.title('Distance VS Number of Generations')
    plt.grid(linestyle='dotted')
    plt.plot(progress,color='#235dba')
    plt.ylabel('Distance')
    plt.xlabel('Generation')
    plt.savefig('../graphs/tsp-1-gen.png',dpi=299)
    plt.show()
def Plot(progress,n):
    plt.title('Distance VS Number of K')
    plt.grid(linestyle='dotted')
    plt.plot(progress,color='#235dba')
    plt.ylabel('Distance')
    plt.xlabel('Reset Number')
    plt.savefig('../graphs/tsp-'+n+'.png',dpi=299)
    plt.show()

cityList = []
for i in range(0,25):
    cityList.append(City(x=int(random.random() * 200), y=int(random.random() * 200))) 






""" Main Code """
main_pop =  initialPopulation(100,cityList)
main_pop_temp = main_pop
print 'Genetic Algorithm: '
print '--------------------------'
geneticAlgorithm(population=cityList, popSize=100, eliteSize=20, mutationRate=0.01, generations=500)
print '--------------------------'
print 'Random Hill Search: '
print '--------------------------'
progress = randomHillSearch(cityList,100)
print '--------------------------'
print 'Simulated Annealing: '
print '--------------------------'
progress = simulatedAnnealing(cityList,100)


