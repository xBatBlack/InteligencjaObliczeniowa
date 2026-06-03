import pygad
import math

def endurance(x, y, z, u, v, w):
    return math.exp(-2*(y-math.sin(x))**2) + math.sin(z*u) + math.cos(v*w)

def fitness_func(ga_instance, solution, solution_idx):
    return endurance(solution[0], solution[1], solution[2], solution[3], solution[4], solution[5])

ga_instance = pygad.GA(
    num_generations=100,
    num_parents_mating=10,
    fitness_func=fitness_func,
    sol_per_pop=50,
    num_genes=6, # 6 metali
    gene_space={'low': 0.0, 'high': 0.99999}, # liczby [0, 1)
    gene_type=float,
    parent_selection_type="sss",
    keep_parents=2,
    crossover_type="single_point",
    mutation_type="random",
    mutation_percent_genes=17 # 17% z 6 genów to ~1 gen (brak warningu)
)

ga_instance.run()

solution, fitness, _ = ga_instance.best_solution()
print(f"Proporcje metali (x, y, z, u, v, w): {solution}")
print(f"Najlepsza znaleziona wytrzymałość stopu: {fitness}")

ga_instance.plot_fitness()