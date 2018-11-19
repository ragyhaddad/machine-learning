#!/usr/bin/python3.5
# -*-coding:Utf-8 -*

import random
import operator
import time
import matplotlib.pyplot as plt
import string 

temps1 = time.time()

#genetic algorithm function
def fitness (password, test_word):
	score = 0
	i = 0
	while (i < len(password)):
		if (password[i] == test_word[i]):
			score+=1
		i+=1
	return score * 100 / len(password)

def generateAWord (length):
	i = 0
	result = ""
	while i < length:
		letter = chr(97 + int(26 * random.random()))
		result += letter
		i +=1
	return result

def generateFirstPopulation(sizePopulation, password):
	population = []
	i = 0
	while i < sizePopulation:
		population.append(generateAWord(len(password)))
		i+=1
	return population

def computePerfPopulation(population, password):
	populationPerf = {}
	for individual in population:
		populationPerf[individual] = fitness(password, individual)
	return sorted(populationPerf.items(), key = operator.itemgetter(1), reverse=True)

def selectFromPopulation(populationSorted, best_sample, lucky_few):
	nextGeneration = []
	for i in range(best_sample):
		nextGeneration.append(populationSorted[i][0])
	for i in range(lucky_few):
		nextGeneration.append(random.choice(populationSorted)[0])
	random.shuffle(nextGeneration)
	return nextGeneration

def createChild(individual1, individual2):
	child = ""
	for i in range(len(individual1)):
		if (int(100 * random.random()) < 50):
			child += individual1[i]
		else:
			child += individual2[i]
	return child

def createChildren(breeders, number_of_child):
	nextPopulation = []
	for i in range(len(breeders)/2):
		for j in range(number_of_child):
			nextPopulation.append(createChild(breeders[i], breeders[len(breeders) -1 -i]))
	return nextPopulation

def mutateWord(word):
	index_modification = int(random.random() * len(word))
	if (index_modification == 0):
		word = chr(97 + int(26 * random.random())) + word[1:]
	else:
		word = word[:index_modification] + chr(97 + int(26 * random.random())) + word[index_modification+1:]
	return word
	
def mutatePopulation(population, chance_of_mutation):
	for i in range(len(population)):
		if random.random() * 100 < chance_of_mutation:
			population[i] = mutateWord(population[i])
	return population

def nextGeneration (firstGeneration, password, best_sample, lucky_few, number_of_child, chance_of_mutation):
	 populationSorted = computePerfPopulation(firstGeneration, password)
	 nextBreeders = selectFromPopulation(populationSorted, best_sample, lucky_few)
	 nextPopulation = createChildren(nextBreeders, number_of_child)
	 nextGeneration = mutatePopulation(nextPopulation, chance_of_mutation)
	 return nextGeneration

def multipleGeneration(number_of_generation, password, size_population, best_sample, lucky_few, number_of_child, chance_of_mutation):
	historic = []
	historic.append(generateFirstPopulation(size_population, password))
	for i in range (number_of_generation):
		historic.append(nextGeneration(historic[i], password, best_sample, lucky_few, number_of_child, chance_of_mutation))
	return historic

#print result:
def printSimpleResult(historic, password, number_of_generation): #bestSolution in historic. Caution not the last
	result = getListBestIndividualFromHistorique(historic, password)[number_of_generation-1]
	print ("solution: \"" + result[0] + "\" fitness: " + str(result[1]))

#analysis tools
def getBestIndividualFromPopulation (population, password):
	return computePerfPopulation(population, password)[0]

def getListBestIndividualFromHistorique (historic, password):
	bestIndividuals = []
	for population in historic:
		bestIndividuals.append(getBestIndividualFromPopulation(population, password))
	return bestIndividuals

#graph
def evolutionBestFitness(historic, password):
	plt.axis([0,len(historic),0,105])
	plt.title("Word Guessed: " + password)
	plt.grid(linestyle='dotted')
	evolutionFitness = []
	for population in historic:
		evolutionFitness.append(getBestIndividualFromPopulation(population, password)[1])
	plt.plot(evolutionFitness)
	plt.ylabel('fitness best individual')
	plt.xlabel('generation')
	plt.show()

def evolutionAverageFitness(historic, password, size_population):
	plt.axis([0,len(historic),0,105])
	plt.title(password)
	plt.grid(linestyle='dotted')
	evolutionFitness = []
	for population in historic:
		populationPerf = computePerfPopulation(population, password)
		averageFitness = 0
		for individual in populationPerf:
			averageFitness += individual[1]
		evolutionFitness.append(averageFitness/size_population)
	plt.plot(evolutionFitness)
    
	plt.ylabel('Average fitness')
	plt.xlabel('generation')
    
	plt.show()

""" Random Hill Restart to crack a password """
def randhomHillSearch():
    print 'Optimizing Please Wait....'
    progress = []
    temp_fit = 0
    best_word = ""
    for x in range(1000000):
        new_word = generateAWord(len(password))
        new_fitness =  fitness(password,new_word)
        if new_fitness > temp_fit:
            temp_fit = new_fitness 
            best_word = new_word 
        progress.append(temp_fit)
    print best_word
    print 'Fitness ',  temp_fit
    return progress

   

""" Simulated Annealing to crack a password """
def simulatedAnnealing():
    print 'Optimizing Please Wait....'
    temp_fit = 0
    best_word = "" 
    progress = []
    new_word = list(generateAWord(len(password)))
    new_fitness =  fitness(password,new_word)
    Temperature_min = 0.000001
    Temperature = 5
    alpha = 0.99
    k = 300
    while Temperature > Temperature_min:
        i = 1
        while i <= k:
            new_char = chr(97 + int(26 * random.random()))
            temp_word = new_word 
            new_word[random.randint(0,len(new_word)-1)] = str(new_char)
            new_word_str = "".join(new_word)
            new_fitness = fitness(password,new_word_str) 
            if new_fitness > temp_fit:
                temp_fit = new_fitness 
                best_word = new_word_str
            else:
                # if random.randint(0,100) * Temperature > 1:
                #     new_word = temp_word
                pass
            progress.append(temp_fit)
            i += 1 
            
        Temperature = Temperature * alpha 

    print best_word
    print 'Fitness ',  temp_fit
    return progress



#variables
password = "batman"
size_population = 100
best_sample = 20
lucky_few = 20
number_of_child = 5
number_of_generation = 50
chance_of_mutation = 5

print 'Password to Crack: ',password

print 'Running Random Hill Search:'
print '--------------------------'
""" RHC """
progress_1 = randhomHillSearch()
print '--------------------------'
print 'Running Simulated Annealing'
print '--------------------------'
""" Simulated Annealing """
progress_2 = simulatedAnnealing()
print '--------------------------'
print 'Running Genetic Algorithm'
print '--------------------------'
print 'Optimizing Please Wait....'
""" Genetic Algo """
#program
if ((best_sample + lucky_few) / 2 * number_of_child != size_population):
	print ("population size not stable")
else:
	historic = multipleGeneration(number_of_generation, password, size_population, best_sample, lucky_few, number_of_child, chance_of_mutation)
	
	printSimpleResult(historic, password, number_of_generation)
	
	# evolutionBestFitness(historic, password)
	# evolutionAverageFitness(historic, password, size_population)

	

