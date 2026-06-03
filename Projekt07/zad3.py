import pygad
import numpy as np
import time

maze = np.array([
    [0,0,0,0,0,0,0,0,0,0,0,0],
    [0,1,1,0,1,1,1,0,1,0,1,0], # (1,1) to Start
    [0,0,1,1,1,0,1,1,0,1,0,0],
    [0,1,0,1,0,0,1,1,0,1,0,0],
    [0,1,1,0,0,1,1,1,0,1,1,0],
    [0,1,1,1,1,0,1,1,1,0,1,0],
    [0,1,0,1,1,0,0,1,1,1,0,0],
    [0,1,0,1,0,0,1,0,0,1,1,0],
    [0,1,0,1,0,1,1,0,1,0,1,0],
    [0,1,0,1,1,1,0,1,0,1,1,0],
    [0,1,0,1,1,1,1,1,1,1,1,0], # (10,10) to Koniec
    [0,0,0,0,0,0,0,0,0,0,0,0]
])

start_pos = (1, 1)
end_pos = (10, 10)

def fitness_func(_ga_instance, solution, _solution_idx):
    x, y = start_pos
    
    for ruch in solution:
        nx, ny = x, y
        
        if ruch == 0: nx -= 1   
        elif ruch == 1: ny += 1 
        elif ruch == 2: nx += 1 
        elif ruch == 3: ny -= 1 
        
        if maze[nx][ny] == 1:
            x, y = nx, ny
            
        if (x, y) == end_pos:
            return 1000
            
    dystans = abs(end_pos[0] - x) + abs(end_pos[1] - y)
    return 1 / (dystans + 1)

gene_space = [0, 1, 2, 3]

sukcesy = 0
czasy = []

print("Rozpoczynam testowanie algorytmu: ")

for i in range(10):
    ga_instance = pygad.GA(
        num_generations=500,           
        num_parents_mating=20,         
        fitness_func=fitness_func,
        sol_per_pop=100,               
        num_genes=30,                  
        gene_space=gene_space,
        parent_selection_type="sss",
        keep_parents=5,
        crossover_type="single_point",
        mutation_type="random",
        mutation_percent_genes=10,
        stop_criteria=["reach_1000"]
    )

    start_time = time.time()
    ga_instance.run()
    end_time = time.time()

    solution, fitness, _ = ga_instance.best_solution()

    if fitness == 1000:
        sukcesy += 1
        czasy.append(end_time - start_time)
        print(f"Próba {i+1}: SUKCES (Czas: {end_time - start_time:.4f} s)")
    else:
        print(f"Próba {i+1}: PORAŻKA (Najlepszy dystans od mety: {(1/fitness)-1})")

print("\n--- PODSUMOWANIE ---")
print(f"Skuteczność: {sukcesy/10 * 100}%")
if czasy:
    print(f"Średni czas działania dla udanych prób: {np.mean(czasy):.4f} sekund")
else:
    print("Nie udało się znaleźć wyjścia w żadnej z prób. Spróbuj zwiększyć sol_per_pop lub num_generations.")