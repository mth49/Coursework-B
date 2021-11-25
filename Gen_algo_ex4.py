import random
from typing import List
import numpy as np
import matplotlib.pyplot as plt
import copy


#auxillary Ga operations
def _utils_constraints(g, min, max):
    if max and g > max:
        g = max
    if min and g < min:
        g = min
        return g
#Blend crossover to choose random genes in the range defined by the parent's genes
def crossover_blend(g1, g2, alpha, min=None, max=None):
    shift = (1. + 2. * alpha) * random.random() - alpha
    new_g1 = (1. - shift)* g1 + shift*g2
    new_g2 = shift * g1 + (1. - shift) * g2

    return _utils_constraints(new_g1, min, max), _utils_constraints(new_g2, min, max)

def mutate_gaussian(g, mu, sigma, min = None, max = None):
    mutated_gene = g + random.gauss(mu, sigma)
    return _utils_constraints(mutated_gene, min, max)

def select_tournament(population, tournament_size):
    new_offspring = []
    for _ in range(len(population)):
        candidates = [random.choice(population) for _ in range(tournament_size)]
        new_offspring.append(max(candidates, key = lambda ind: ind.fitness))
    return new_offspring

def func(x):
    return 5*x**5+18*x**4+31*x**3-14*x**2+7*x+19

def get_best(population):
    best = population[0]
    for ind in population:
        if ind.fitness > best.fitness:
            best = ind
    return best
#plot population over generations
def plot_population(population, number_of_population):
    best = get_best(population)
    x = np.linspace(-100,100)
    plt.plot(x, func(x), '--', color='blue')
    plt.plot([ind.get_gene() for ind in population], [ind.fitness for ind in population], 'o', color = 'orange')
    plt.plot([best.get_gene()], [best.fitness], 's', color = 'green')
    plt.title(f"Generation number {number_of_population}")
    plt.show()
    plt.close()

#individual class
class Individual:

    def __init__(self, gene_list: List[float]) -> None:
        self.gene_list =gene_list
        self.fitness = func(self.gene_list[0])
    
    def get_gene(self):
        return self.gene_list[0]
    
    @classmethod
    def crossover(cls, parent1, parent2):
        child1_gene, child2_gene =crossover_blend(parent1.get_gene(), parent2.get_gene(), 1, -10, 10)
        return Individual([child1_gene]), Individual([child2_gene])

    
    @classmethod
    def mutate(cls, ind):
        mutated_gene = mutate_gaussian(ind.gen_gene(), 0, 1, -10, 10)
        return Individual([mutated_gene])
    
    @classmethod
    def select(cls, population):
        return select_tournament(population, tournament_size = 3)
        #tournament selection, subgroup selected from population 

    @classmethod
    def create_random(cls):
        return Individual([random.randrange(-1000, 1000) / 100])
##Ga flow
random.seed(52)
#
POPULATION_SIZE = 10
CROSSOVER_PROBABILITY = 0.8
MUTATION_PROBABILITY = 0.1
MAX_GENERATIONS = 100

first_population = [Individual.create_random() for _ in range(POPULATION_SIZE)]
plot_population(first_population, 0)

generation_number = 0

population = first_population.copy()

while generation_number < MAX_GENERATIONS:
    generation_number += 1
    #Selection
    offspring = Individual.select(population)

    #Crossover, use zip to pass both values of touples together as it passes through the for loop
    crossed_offspring = []
    for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
        if random.random() < CROSSOVER_PROBABILITY:
            kid1, kid2 = Individual.crossover(ind1, ind2)
            crossed_offspring.append(kid1)
            crossed_offspring.append(kid2)
        else:
            crossed_offspring.append(ind1)
            crossed_offspring.append(ind2)
    
    #Mutation
    mutated_offspring = []
    for mutant in crossed_offspring:
        if random.random() < MUTATION_PROBABILITY:
            new_mutant = Individual.mutate(mutant)
            mutated_offspring.append(new_mutant)
        else:
            mutated_offspring.append(mutant)
    population = mutated_offspring.copy()
    plot_population(population, generation_number)


