import torch
import random

# Initialize Parameters
popSize = 100  # Number of individuals in the population
bin_len = 5  # Number of binary genes in each chromosome
cont_len = 5  # Number of continuous genes in each chromosome
chr_len = bin_len + cont_len  # Total chromosome length
mutation_rate = 0.01  # Probability of mutation in each gene
gens = 100  # Number of generations to evolve

# Sum of binary values + sum of continuous values
def fitness(chromosome):
    binSum = torch.sum(chromosome[:bin_len])
    contSum = torch.sum(chromosome[bin_len:])
    return binSum + contSum

# Initialize population with random binary and continuous values
def initialize_population():
    bin = torch.randint(0, 2, (popSize, bin_len)).float()
    cont = torch.rand(popSize, cont_len)
    return torch.cat((bin, cont), dim=1)

# Select parents based on fitness using a probability proportional to fitness
def select_parents(population, fit):
    prob = fit / torch.sum(fit)
    parent_idx = torch.multinomial(prob, popSize, replacement=True)
    return population[parent_idx]

# Single-point crossover to create offspring
def crossover(parents):
    offspring = torch.zeros_like(parents)
    for i in range(0, popSize, 2):
        parent1, parent2 = parents[i], parents[i+1]
        crossover_point = random.randint(1, chr_len - 1)
        cpt=crossover_point
        offspring[i, :cpt] = parent1[:cpt]
        offspring[i, cpt:] = parent2[cpt:]
        offspring[i+1, :cpt] = parent2[:cpt]
        offspring[i+1, cpt:] = parent1[cpt:]
    return offspring

# Mutation: flip random binary bits and mutate continuous values
def mutate(offspring):
    binMut = torch.rand_like(offspring[:, :bin_len]) < mutation_rate
    contMut = torch.rand_like(offspring[:, bin_len:]) < mutation_rate

    offspring[:, :bin_len][binMut] = 1 - offspring[:, :bin_len][binMut]
    offspring[:, bin_len:][contMut] = torch.rand(torch.sum(contMut))
    
    return offspring

# Main genetic algorithm
def genetic_algorithm():
    population = initialize_population()
    for generation in range(gens):
        fit = torch.tensor([fitness(individual) for individual in population])
        parents = select_parents(population, fit)
        offspring = crossover(parents)
        population = mutate(offspring)
    return population

# Run everything
finalPop = genetic_algorithm()
best = max(finalPop, key=fitness)
stats = {
    f'Binary Gene {i+1}': gene.item() for i, gene in enumerate(best[:bin_len])
}
stats.update({
    f'Continuous Gene {i+1}': gene.item() for i, gene in enumerate(best[bin_len:])
})

print(stats, fitness(best).item())
