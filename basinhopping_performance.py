import numpy as np
from scipy.optimize import basinhopping

'''Run file to get basinhopping performance, Note 50 repeats takes a very long time!'''

def objective_function(x):
    for x_i in x: 
        if abs(x_i) > 500.0 : return np.inf

    return np.sum(-x*np.sin(np.sqrt(abs(x))))


repeats = 50


solutions = []
best_solution = np.inf
best_x = None


for i in range(repeats):
    x = np.random.uniform(-500,500,6)


    optimal = basinhopping(objective_function,x,15_000)
    x,res = optimal.x,optimal.fun

    solutions.append(res)
    
    if res < best_solution:
        best_x = x
        best_solution = res


    print(f'{i+1} out of {repeats} finished!')
    print(x,res)

print('-'*162)
print('')
print(f'Best Solution for scipy.optimize basinhopping method at 15,000 objective function evaluations')
print('')

print(f'x = {best_x}')
print(f'objective function = {best_solution}')
print(f'Mean of solutions {np.mean(solutions)}')
print(f'Standard deviation of solutions {np.std(solutions)}')
print('')
print('-'*162)
print('')
