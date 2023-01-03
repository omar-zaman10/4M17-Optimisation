import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
from pylab import meshgrid
from progress_bar import progress_bar



class ES_Solver:
    '''Evolutionary Strategies Solver object
    
    intitialise() : reinitialises the solver object for repeated use

    set_population_offspring_size(pop_size,offspring_size): sets the population and offspring sizes

    set_elitist(elitist): sets the selection scheme used

    set_recombinations(recombinations): sets the recombination scheme used

    objective_function(x): returns Schwefels function for a solution x

    out_of_bounds(x): checks if solution is out of bounds 

    Two choices of selection schemes used in the solver
    selection():
    elitist_selection():

    3 choices of recombination schemes used in the solver
    discrete_recombinations():
    global_discrete_recombinations():
    intermediate_recombinations(w=0.5):

    Mutation and updating strategy parameters
    mutate():
    update_cov():

    solve(): Performs the Evolutionary Strategies algorithm with the given recombinations, selection scheme and population size

    Methods for plotting the data after solve() has been performed
    plot()
    plot_2d()
    '''

    def __init__(self,d=6):
        '''Initialising the ES_Solver with the population, offspring and choices for recombinations and selection'''
        self.d = d #dimensions
        self.offspring_size = 500 #500
        self.population_size = 200 #200
        self.recombinations = 'global discrete'
        self.elitist = False
        self.counter = 0
        self.max_iter = 15_000 
        self.covariance =  np.diag(500*np.ones(d))

        self.population = [np.random.uniform(-500,500,d) for _ in range(self.population_size)] #List of np arrays
        self.offspring = [] #Should be []


        self.data = [] #Has all the populations over time

        self.best = np.inf
        self.x = None
       
    def initialise(self):
        '''Reinitialises the solver for another execution of the solver'''

        self.__init__(self.d)

    def set_population_offspring_size(self,pop_size,offspring_size):
        '''Setting the population and offspring size, both arguments have to be int
        pop_size: int population size
        offspring_size: int offspring size'''
        self.population_size = pop_size
        self.offspring_size = offspring_size
    
    def set_elitist(self,elitist):
        '''Setting if elitist selection will be used or not,
         elitist : Boolean'''
        self.elitist = elitist

    def set_recombinations(self,recombinations):
        '''Setting which type of recombinations are used, recombinations : string 3 choices listed below'''
        if recombinations != 'global discrete' and recombinations != 'discrete' and recombinations != 'intermediate':
            raise ValueError('recombinations have to be given as "global discrete","discrete" or "intermediate"')

        self.recombinations = recombinations

    def objective_function(self,x):
        '''Schwefels objective function which has built in constraints,
         returns the objective value for a given data point x'''
        for x_i in x: 
            if abs(x_i) > 500.0 : return np.inf

        return np.sum(-x*np.sin(np.sqrt(abs(x))))
    
    def out_of_bounds(self,x):
        '''Checks if solution x violates any constraints
        return type Boolean'''
        for x_i in x:
            if abs(x_i) > 500: return True
        return False

    def selection(self):
        '''(mu,lambda) selection scheme for updating the population from the generated offspring
        returns list of best solutions'''
        self.offspring.sort(key=lambda x:self.objective_function(x))
        self.counter += self.offspring_size
        return self.offspring[:self.population_size]

    def elitist_selection(self):
        '''(mu+lambda) selection which allows the best solution to survive over time - elitist
        returns list of best solutions'''
        p = self.population + self.offspring
        p.sort(key=lambda x:self.objective_function(x))
        self.counter += self.offspring_size
        return p[:self.population_size]

    def global_discrete_recombinations(self):
        '''Recombinations which generate the offpring based on global discrete method'''
        self.offspring = []
        for _ in range(self.offspring_size):
            child = []
            for i in range(self.d):
                child.append(self.population[random.randint(0,self.population_size-1)][i])

            
            self.offspring.append(np.array(child))

    def discrete_recombinations(self):
        '''Recombinations which generate offspring based on discrete method between 2 randomly selected parents'''
        self.offspring = []
        for _ in range(self.offspring_size):
            child = []

            parents = random.choices(self.population,k=2)
            for i in range(self.d):
                child.append(parents[random.randint(0,1)][i])

            self.offspring.append(np.array(child))  

    def intermediate_recombinations(self,w=0.5):
        '''Intermediate recombination scheme which generates offspring 
        based on mean values of a pair of randomly selected parents'''
        self.offspring = []

        for _ in range(self.offspring_size):
            

            parents = random.choices(self.population,k=2)
            child = w*parents[0] + (1-w)*parents[1]

            self.offspring.append(np.array(child)) 

    def update_cov(self):
        '''Updates the covariance matrix that allows for maximum mutation in each dimension'''
        tau = 1.0/(np.sqrt(2 * np.sqrt(self.d)))
        taud = 1.0/(np.sqrt(2*self.d))
        N0 = np.random.normal()
        Nis = np.array([np.random.normal() for _ in range(self.d)])

        variances = np.diag(self.covariance)
        variances = variances**0.5 * np.exp(taud*N0 +tau*Nis)
        variances = variances**2
        variances = np.clip(variances,0,500)

        self.covariance = np.diag(variances)
    
    def mutate(self):
        '''Mutation of current population using the covariance matrix'''
        self.update_cov()

        for i,x in enumerate(self.population):
            n = np.random.multivariate_normal(mean=np.zeros(self.d),cov = self.covariance)
            x_dash = x+n


            while self.out_of_bounds(x_dash):
                n = np.random.multivariate_normal(mean=np.zeros(self.d),cov = self.covariance)
                x_dash = x+n
            
            self.population[i] = x_dash

    def solve(self):
        '''Performs the Evolutionary Strategy Algorithm and minimises the objective function
        returns Dict 
        'fun' : value for best soltion
        'x' : best solution '''

        while self.counter < self.max_iter:
            #Mutation
            self.mutate()

            #Recombination
            if self.recombinations == 'global discrete':
                self.global_discrete_recombinations()
            elif self.recombinations == 'discrete':
                self.discrete_recombinations()
            elif self.recombinations == 'intermediate':
                self.intermediate_recombinations()


            #Selection stage
            if not self.elitist:
                self.population = self.selection()

            else: 
                self.population  = self.elitist_selection()

            value = self.objective_function(self.population[0])

            if value < self.best:
                self.best = value
                self.x = self.population[0]

            #progress_bar(self.counter,self.max_iter)
            self.data.append(self.population[:])


        return {'x':self.x,'fun':self.best}

    def plot(self):
        '''Used to see plots of population means and best solutions over the generations'''

        sns.set_style('darkgrid')
        population_value = []
        for pop in self.data:
            values = [np.sum(-x*np.sin(np.sqrt(abs(x)))) for x in pop]
            population_value.append(values[:])
        means = [np.mean(pop) for pop in population_value]
        bests = [self.objective_function(pop[0]) for pop in self.data]

        plt.plot(means,c='b',label='population mean')
        plt.plot(bests,c='r',label='population best')
        plt.xlabel('Generations')
        plt.ylabel('Objective function')
        plt.legend()
        plt.show()

    def plot2d(self):
        '''Used to plot how generations change overtime, only a few of the first 10 generations are plotted to see
        the evolution'''
        if self.d != 2: raise ValueError(f'Only available for 2d')
        sns.set_style('whitegrid')


        x = np.linspace(-500,500,1000)
        y = np.linspace(-500,500,1000)
        X,Y = meshgrid(x,y)
        Z = -X*np.sin(np.sqrt(abs(X))) -Y*np.sin(np.sqrt(abs(Y)))
        cmap = 'gist_earth'


        for i,pop in enumerate(self.data[0:10:2]):

            plt.figure(figsize=(8,6))
            plt.contour(X, Y, Z, 15, cmap=cmap)
            plt.xlabel('x1')
            plt.ylabel('x2')
            plt.title(f'Generation {i*2+1}')


            data1 = [x[0] for x in pop]
            data2 = [x[1] for x in pop]
            sc = plt.scatter(data1,data2,marker='o',s=50,c='darkred',edgecolors='black')
            plt.savefig(f'Generation {i*2+1}')
            plt.show()

        pop = self.data[-1]
        plt.figure(figsize=(8,6))
        plt.contour(X, Y, Z, 15, cmap=cmap)
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.title('Schwefels function 2D last generation')


        data1 = [x[0] for x in pop]
        data2 = [x[1] for x in pop]
        sc = plt.scatter(data1,data2,marker='o',s=50,c='maroon',edgecolors='black')
        #plt.savefig('testplot2')
        plt.show()

solver = ES_Solver
    

def main(solver,elitist,recombinations,population,offspring,repeats):
    '''
    Runs repeated trials of the ES algorithm
    mysolver: ES_solver object
    elitist:  Boolean of whether elitist selection will be used
    recombinations: string specifying which recombination choice 'discrete' 'global discrete' 'intermediate'
    population: Population size int
    offspring: Offspring size int
    repeats: Number of repeated trials
    
    prints out the results on the terminal
    '''
    best_solution = np.inf
    best_x = None
    solutions = []

    s = 'normal'
    if elitist: s = 'elitist'


    for i in range(repeats):

        solver.initialise()
        solver.set_elitist(elitist)
        solver.set_recombinations(recombinations)
        solver.set_population_offspring_size(population,offspring)
        res = solver.solve()

        x,val = res['x'],res['fun']


        if val < best_solution:
            best_x = x
            best_solution = val


        solutions.append(val)
        progress_bar(i+1,repeats)

    print('-'*162)
    print('')
    print(f'Best Solution for {s} selection with {recombinations} recombinations and (population,offspring) size of {population,offspring}')
    print('')
    print(f'x = {best_x}')
    print(f'objective function = {best_solution}')
    print(f'Mean of solutions {np.mean(solutions)}')
    print(f'Standard deviation of solutions {np.std(solutions)}')
    print('')
    print('-'*162)
    print('')


if __name__ == '__main__':


    solver = ES_Solver(6)
    

    #Discrete recombinations
    main(solver,False,'discrete',30,200,50)

    main(solver,True,'discrete',30,200,50)

    main(solver,False,'discrete',200,500,50)

    main(solver,True,'discrete',200,500,50)
    

    #Global Discrete recombinations
    main(solver,False,'global discrete',30,200,50)

    main(solver,True,'global discrete',30,200,50)

    main(solver,False,'global discrete',200,500,50)

    main(solver,True,'global discrete',200,500,50)


    #Intermediate recombinations
    main(solver,False,'intermediate',30,200,50)

    main(solver,True,'intermediate',30,200,50)

    main(solver,False,'intermediate',200,500,50)

    main(solver,True,'intermediate',200,500,50)
