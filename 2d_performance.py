from simulated_annealing import SA_Solver
from evolutionary_strategies import ES_Solver
import numpy as np

'''
Run this file to get the 2D search space performances of the Simulated Annealing and Evolutionary Strategies algorithms
'''



my_solver = SA_Solver(2)
res = my_solver.solve()
x,value ,archive = res['x'], res['fun'], res['archive']
print(f'x {x} objective {value}')


my_solver.plot()
my_solver.plot_2d()


solver = ES_Solver(2)


res = solver.solve()
x,value = res['x'],res['fun']
print(f'x {x} objective {value}')

solver.plot2d()
