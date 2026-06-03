import pygad
import numpy as np
import time

przedmioty = ["zegar", "obraz-pejzaż", "obraz-portret", "radio", "laptop", 
              "lampka nocna", "srebrne sztućce", "porcelana", "figura z brązu", 
              "skórzana torebka", "odkurzacz"]
wartosci = np.array([100, 300, 200, 40, 500, 70, 100, 250, 300, 280, 300])
wagi = np.array([7, 7, 6, 2, 5, 6, 1, 3, 10, 3, 15])
limit_wagi = 25

def fitness_func(ga_instance, solution, solution_idx):
    waga_suma = np.sum(solution * wagi)
    wartosc_suma = np.sum(solution * wartosci)
    
    if waga_suma > limit_wagi:
        return 0 # Kara za przekroczenie wagi
    return wartosc_suma

sukcesy = 0
czasy = []

for _ in range(10):
    ga_instance = pygad.GA(
        num_generations=100, #max liczba pokoleń
        num_parents_mating=10, #ile najlepszych rodziców do krzyżowania
        fitness_func=fitness_func,
        sol_per_pop=50,
        num_genes=len(przedmioty),
        gene_space=[0, 1],
        parent_selection_type="sss",
        keep_parents=2,
        crossover_type="single_point",
        mutation_type="random",
        mutation_percent_genes=10,
        stop_criteria=["reach_1630"]
    )
    
    start = time.time()
    ga_instance.run()
    end = time.time()
    
    solution, fitness, _ = ga_instance.best_solution()
    
    if fitness == 1630:
        sukcesy += 1
        czasy.append(end - start)

print(f"Skuteczność: {sukcesy/10 * 100}%")
if czasy:
    print(f"Średni czas udanego szukania: {np.mean(czasy):.4f} sekund")

rozwiazanie, ocena, _ = ga_instance.best_solution()
print(f"Najlepsza wartość: {ocena}")
zabierane = [przedmioty[i] for i in range(len(przedmioty)) if rozwiazanie[i] == 1]
print(f"Zabierane przedmioty: {zabierane}")
ga_instance.plot_fitness()