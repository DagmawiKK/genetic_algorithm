import random
import numpy as np
import math
from hyperon import MeTTa, ValueAtom, ExpressionAtom, GroundedAtom, Atom, E
POP_SIZE = 10
GENES = 8
GENERATIONS = 30
ELITE_COUNT = 3
CROSSOVER_RATE = 0.9
MUTATION_RATE = 0.1
INITIAL_MUTATION_STD = 0.5  # High initial value
MUTATION_DECAY = 0.95       # Decay factor per generation
INITIAL_SBX_ETA = 2
SBX_ETA_GROWTH = 1.05       # SBX eta increases each generation

# === Fixed input individuals for blend ===
INPUT_A = [0.9, 0.9, 0.0, 0.8, 0.2, 0.9, 0.7, 0.7]
INPUT_B = [0.8, 0.2, 1.0, 0.3, 0.9, 0.4, 0.6, 0.3]

metta = MeTTa()
with open("metta_modules/helper_functions.metta") as f:
    helper = f.read()
metta.run(helper)

with open("metta_modules/initialize_population.metta") as f:
    initializer = f.read()
metta.run(initializer)

with open("metta_modules/fitness.metta") as f:
    fitness_code = f.read()
metta.run(fitness_code)

with open("metta_modules/roulette_stochastic_acceptance.metta") as f:
    roulette_code = f.read()
metta.run(roulette_code)

with open("metta_modules/sbx.metta") as f:
    sbx_code = f.read()
metta.run(sbx_code)

with open("metta_modules/mutate.metta") as f:
    mutate_code = f.read()
metta.run(mutate_code)

def return_value(result):
    res = []
    for r in result[0]:
        a = r.get_children()
        for b in a:
            if isinstance(b, GroundedAtom):
                res.append(b.get_object().value)
            else:
                res.append(b)
    return res
    

# === Fitness Function ===
def fitness(candidate):
    # emergence = [c - max(a, b) for c, a, b in zip(candidate, INPUT_A, INPUT_B)]
    # emergence = [max(0, e) for e in emergence]  # clamp negative emergence to 0
    # contributions = [min(a, b) * e for a, b, e in zip(INPUT_A, INPUT_B, emergence)]
    # total = sum(contributions)
    # return min(total / GENES, 1.0)
    individual_str = " ".join(map(str, candidate))
    input_a = " ".join(map(str, INPUT_A))
    input_b = " ".join(map(str, INPUT_B))
    metta_code = f"!(fitness ({individual_str}) ({input_a}) ({input_b}) {GENES})"
    
    result = metta.run(metta_code)
    return list(*result)[0].get_object().value

def initialize_population():
    result = metta.run(f"!(make-pop {POP_SIZE} {GENES})")
    for r in result[0]:
        result = r.get_children()
    res = []
    for r in result:
        a = r.get_children()
        t = []
        for b in a:
            if isinstance(b, GroundedAtom):
                t.append(b.get_object().value)
            else:
                t.append(b)
        res.append(t)
    return res

# === Selection: Roulette-Wheel via Stochastic Acceptance ===

def roulette_wheel_selection_T(population, fitnesses):
    total_fitness = sum(fitnesses)
    
    spin = random.uniform(0, total_fitness)
    
    current_sum = 0
    for i, (ind, fit) in enumerate(zip(population, fitnesses)):
        current_sum += fit
        if current_sum >= spin:
            return ind
    
    return population[-1]
def roulette_stochastic_acceptance(population, fitnesses):
    # w_max = max(fitnesses)
    # if w_max == 0:
    #     return random.choice(population)
    # while True:
    #     i = random.randint(0, len(population) - 1)
    #     if random.random() < fitnesses[i] / w_max:
    #         return population[i]
#     !(roulette-stochastic-acceptance
#    ((0.8 1.0 0.8 0.1) (0.0 0.8 0.2) (1.0 0.3 0.9) (0.9 0.4 0.6))
#    (0.1 0.3 0.5 0.1) 4)
    pop = []
    for ind in population:
        pop.append(f"({' '.join(map(str, ind))})")
    population_str = "(" +" ".join(pop) + ")"

    fitnesses_str = "(" + " ".join(map(str, fitnesses)) + ")"
    metta_code = f"!(roulette-stochastic-acceptance {population_str} {fitnesses_str} {POP_SIZE})"
    result = metta.run(metta_code)
    return return_value(result)

# === Crossover: Simulated Binary Crossover (Adaptive Eta) ===
def sbx_crossover(p1, p2, eta):
    # if random.random() > CROSSOVER_RATE:
    #     return p1[:], p2[:]

    # child1, child2 = [], []
    # for x1, x2 in zip(p1, p2):
    #     if random.random() <= 0.5:
    #         if abs(x1 - x2) > 1e-14:
    #             x1, x2 = min(x1, x2), max(x1, x2)
    #             rand = random.random()
    #             beta = 1.0 + (2.0 * (x1) / (x2 - x1))
    #             alpha = 2.0 - beta ** -(eta + 1)
    #             if rand <= 1.0 / alpha:
    #                 betaq = (rand * alpha) ** (1.0 / (eta + 1))
    #             else:
    #                 betaq = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta + 1))
    #             c1 = 0.5 * ((x1 + x2) - betaq * (x2 - x1))
    #             c2 = 0.5 * ((x1 + x2) + betaq * (x2 - x1))
    #             child1.append(min(max(c1, 0.0), 1.0))
    #             child2.append(min(max(c2, 0.0), 1.0))
    #         else:
    #             child1.append(x1)
    #             child2.append(x2)
    #     else:
    #         child1.append(x1)
    #         child2.append(x2)
    # print(f"SBX Crossover")
    # return child1, child2
    p1_str = " ".join(map(str, p1))
    p2_str = " ".join(map(str, p2))
    metta_code = f"!(sbx-crossover ({p1_str}) ({p2_str}) {eta} {CROSSOVER_RATE})"
    result = metta.run(metta_code)
    tmp = return_value(result)
    child1 = [child.get_object().value for child in tmp[0].get_children()]
    child2 = [child.get_object().value for child in tmp[2].get_children()]
    return child1, child2
    


# === Mutation ===
def mutate(individual, mutation_std):
    # for i in range(len(individual)):
    #     if random.random() < MUTATION_RATE:
    #         individual[i] += random.gauss(0, mutation_std)
    #         individual[i] = min(max(individual[i], 0), 1)  # clip to [0, 1]
    # return individual
    individual_str = " ".join(map(str, individual))
    metta_code = f"!(mutate ({individual_str}) {mutation_std} {MUTATION_RATE})"
    result = metta.run(metta_code)
    result = return_value(result)
    return result


# === Main GA Loop ===
def genetic_algorithm():
    population = initialize_population()
    mutation_std = INITIAL_MUTATION_STD
    sbx_eta = INITIAL_SBX_ETA
    for gen in range(GENERATIONS):
        fitnesses = []
        for ind in population:
            f = fitness(ind)
            fitnesses.append(f)
        elites = sorted(zip(population, fitnesses), key=lambda x: x[1], reverse=True)[:ELITE_COUNT]
        new_population = [ind for ind, _ in elites]

        while len(new_population) < POP_SIZE:
            parent1 = roulette_stochastic_acceptance(population, fitnesses)
            parent2 = roulette_stochastic_acceptance(population, fitnesses)
            child1, child2 = sbx_crossover(parent1, parent2, eta=sbx_eta)
            new_population.extend([mutate(child1, mutation_std), mutate(child2, mutation_std)])

        population = new_population[:POP_SIZE]  # Ensure population size remains constant
        best_fitness = max(fitnesses)
        print(f"Generation {gen+1}: Best Fitness = {best_fitness:.4f} | Mutation STD = {mutation_std:.4f} | SBX_ETA = {sbx_eta:.2f}")

        mutation_std *= MUTATION_DECAY  # decay mutation over time
        sbx_eta *= SBX_ETA_GROWTH       # increase SBX eta over time (more conservative)

    # Final result
    final_fitnesses = [fitness(ind) for ind in population]
    best = max(zip(population, final_fitnesses), key=lambda x: x[1])
    print("\nBest Individual:", best[0])
    print("Best Fitness:", best[1])

if __name__ == "__main__":
    genetic_algorithm()