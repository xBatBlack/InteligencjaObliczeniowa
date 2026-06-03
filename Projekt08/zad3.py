import numpy as np

# Prosty labirynt: 0 - wolne, 1 - ściana
# S = Start (0,0), E = Wyjście (4,4)
maze = np.array([
    [0, 0, 1, 1, 1],
    [1, 0, 0, 0, 1],
    [1, 1, 1, 0, 1],
    [1, 0, 0, 0, 0],
    [1, 1, 1, 1, 0]
])

def create_distance_matrix(maze):
    rows, cols = maze.shape
    num_nodes = rows * cols
    # Inicjujemy macierz nieskończonościami (brak przejścia)
    dist_matrix = np.full((num_nodes, num_nodes), np.inf)
    
    for r in range(rows):
        for c in range(cols):
            if maze[r, c] == 1:
                continue # Ściana
                
            current_node = r * cols + c
            
            # Sprawdzamy sąsiadów (góra, dół, lewo, prawo)
            directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and maze[nr, nc] == 0:
                    neighbor_node = nr * cols + nc
                    # Odległość do sąsiada to 1
                    dist_matrix[current_node, neighbor_node] = 1
                    
    return dist_matrix

distance_matrix = create_distance_matrix(maze)