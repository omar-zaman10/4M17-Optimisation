import numpy as np
import random
from scipy.optimize import basinhopping
from progress_bar import progress_bar
import matplotlib.pyplot as plt
from pylab import meshgrid
import seaborn as sns

'''Run file to execute SA algorithm with all the different implentations and printed on console will be the results'''

class SA_Solver:
    '''
    SA_Solver object which runs the Simulated Annealing Algorithm

    initialise(): reinitialises solver for repeated use

    set_T0(T0): sets initial temperature T0

    set_adaptive(adaptive): sets the cooling scheme

    set_restarts(use_restart): sets if restarts are allowed

    objective_function(x): returns Schwefels function on a solution x

    acceptance(df): returns acceptance probability on an increase in objective function df

    methods which generate x, update x and update the diagonal matrix D
    gen_new_x():
    update_x()
    update_D(Du,alpha=0.1,w = 2.1):

    methods responsible for generating alpha and updating temperature based on annealing schedule
    annealing_schedule(alpha = 0.95,n_min=50,L_k=200): 
    generate_alpha():

    restart(): Performs the restart

    solve(): Executes the SA algorithm with given initial temperature T0, cooling scheme and usage of restarts

    methods for plotting data after solve() has been executed
    plot()
    plot_2d()
    
    '''


    def __init__(self,d=6):
        '''Initialises the dimensionality of the solver d as well inital paramters x,D,T and creates the archive'''
        self.d = d
        self.x = np.random.uniform(-500,500,d)
        self.D = np.diag(250*np.ones(d)) #Diagonal matrix of max allowable changes
        self.T = 1000 #T0 initial temperature found using intial search Change to np.inf for initial search
        self.final_T = 1 #Final temperature
        self.max_iter = 15_000
        self.counter = 0
        self.prev_count = None
        self.prev_objective = None
        self.acceptances = 0
        self.trials = 0
        self.data = [[],[]] #x coordinates and function value for plotting
        self.adaptive = False 
        self.use_restart = True


        #Archives both the best solution and every df increase thats accepted as well as all the observed objective functions
        #observed {T:[observed objective functions]} T which corresponds to observed objective functions
        self.archive = {'solution':[None,None],'accepted':[],'observed':{self.T:[],self.final_T :[]}} 

    def initialise(self):
        '''Can reinitialise solver when repeated trials are wanted'''
        self.x = np.random.uniform(-500,500,self.d)
        self.D = np.diag(250*np.ones(self.d)) #Diagonal matrix of max allowable changes
        self.T = 1100 #T0 initial temperature found using intial search Change to np.inf for initial search
        self.final_T = 1 #Final temperature
        self.max_iter = 15_000
        self.counter = 0
        self.prev_objective = None
        self.acceptances = 0
        self.trials = 0
        self.data = [[],[]] #data used for plotting 
        self.adaptive = False 
        self.use_restart = True

        #Archives both the best solution and every df increase thats accepted as well as all the observed objective functions
        #observed {T:[observed objective functions]} T which corresponds to observed objective functions
        self.archive = {'solution':[None,None],'accepted':[],'observed':{self.T:[],self.final_T :[]}} 

    def set_T0(self,T0):
        '''Sets initial temperature
        args T0 - float greater than 0'''
        self.T = T0
        self.archive['observed'][self.T] = []

    def set_adaptive(self,adaptive):
        '''Sets whether adaptive temperature decrement is used
        args adapative - Boolean'''
        self.adaptive = adaptive
        
    def set_restarts(self,use_restart):
        '''Sets whether restarts will be used or not
        args use_restart  - Boolean'''
        self.use_restart = use_restart
        
    
    def objective_function(self,x):
        '''Returns Schewefels function with constraints in place for a given trial solution x'''
        for x_i in x: 
            if abs(x_i) > 500.0 : return np.inf

        return np.sum(-x*np.sin(np.sqrt(abs(x))))

    def acceptance(self,df):
        '''Acceptance probability for a given change in objective
        function df'''
        return min(np.exp(-df/self.T),1)


    def gen_new_x(self):
        '''Generates a new x trial solution, returns the new trial soltion and a numpy array
         of the diagonal values for the R matrix needed to update D'''
        u = np.random.uniform(-1.0,1.0,self.d)
        return self.x + self.D.dot(u), abs(self.D.dot(u))


    def update_D(self,Du,alpha=0.1,w = 2.1):
        '''Uses the diagonal elements of Du given by gen_new_x method to create R 
        and update the D matrix after a succesful trial'''
        R = np.diag(Du)
        self.D = (1-alpha)*self.D + alpha*w*R


    def update_x(self):
        '''Generates a new trial solution x and updates x if accepted, updates matrix D '''

        x_new,Du = self.gen_new_x()
        objective_new = self.objective_function(x_new)
        df = objective_new - self.prev_objective

       
        accept = self.acceptance(df)
        if np.isnan(accept) :  accept = 0

        #If accepted update diagonal matrix D, x and update archive if best solution so far
        if random.random() < accept:
            self.archive['observed'][self.T].append(objective_new)
            if df >0: self.archive['accepted'].append(df)

            self.acceptances +=1
            self.update_D(Du)
            self.x = x_new
            self.prev_objective = objective_new

            if self.archive['solution'][1] > self.prev_objective:
                self.archive['solution'] = [self.x,self.prev_objective]
                self.prev_count = self.counter
    
    def annealing_schedule(self,alpha = 0.95,n_min=50,L_k=200):
        '''Updates the temperature by multiplying by alpha after either n_min acceptances
         or L_k trials have occurred'''
        if self.acceptances >= n_min or self.trials >= L_k:

            if self.adaptive: alpha = self.generate_alpha()

            self.acceptances = 0
            self.trials = 0
            self.T *= alpha

            
            self.T = max(self.T,self.final_T)
            if self.T not in self.archive['observed'] : self.archive['observed'][self.T] = []
                

    def generate_alpha(self):
        '''Returns an alpha using an adaptive statistical approach of the current acceptances standard deviation'''

        sigma = np.std(self.archive['observed'][self.T])

        return max(0.5,np.exp(-0.7*self.T / sigma))



    def solve(self):
        '''Method used to run the Simulated annealing algorithm,
        returns dict with the best solution x and its objective function value fun and the archive stored
        'x': coordinates of solution
        'fun : objective function value minimised at x
        'archive': {'solution':objective function, 'accepted': [list of accepted increases of objective function] }
        '''
        self.prev_objective = self.objective_function(self.x)
        self.archive['solution'] = [self.x,self.prev_objective]
        while self.counter < self.max_iter:
            self.update_x()
            
            self.annealing_schedule()
            #progress_bar(self.counter+1,self.max_iter)
            self.counter +=1
            self.trials +=1
            self.data[0].append(self.x)
            self.data[1].append(self.prev_objective)

            #If a long time has occured since previous best solution go back to best solution
            if self.prev_count is not None and self.counter - self.prev_count>3000 and self.use_restart:
                self.restart()

        x,fun = self.archive['solution']
        return {'x':x,'fun':fun,'archive':self.archive}

    def restart(self):
        '''Returns to best solution found so far'''
        self.x,self.prev_objective = self.archive['solution']
        self.prev_count = self.counter
        

    def plot(self):
        '''Plots the objective function vs iteration to see how it minimises over time'''
        sns.set_style('darkgrid')
        plt.figure(figsize=(8,6))
        plt.plot(self.data[1])
        plt.xlabel('Iterations')
        plt.ylabel('Objective function value')
        plt.title('Minimisation of 2D Schwefel function')
        plt.savefig('Simulated Annealing convergence')
        plt.show()

    def plot_2d(self):
        '''Plots the 2D search space exploration across different iterations '''

        if self.d != 2: raise ValueError('Dimensions need to be 2 for this plot')
        sns.set_style('whitegrid')
        plt.figure(figsize=(8,6))
        x = np.linspace(-500,500,1000)
        y = np.linspace(-500,500,1000)
        X,Y = meshgrid(x,y)
        Z = -X*np.sin(np.sqrt(abs(X))) -Y*np.sin(np.sqrt(abs(Y)))
        cmap = 'gist_earth'
        plt.contour(X, Y, Z, 15, cmap=cmap)
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.title('Schwefels function 2D')

        it = [i for i in range(0,self.max_iter,2)]
        data1 = [self.data[0][i][0] for i in it]
        data2 = [self.data[0][i][1] for i in it]
      
        sc = plt.scatter(data1,data2,marker='v',c=it, cmap='brg',s=10)
        plt.colorbar(sc,label='Iterations')
        plt.savefig('Simulated Annealing 2D search pattern')
        plt.show()


def main(my_solver,T0,adaptive,restarts,repeats):
    '''Runs repeated trials of the SA algorithm
    mysolver: SA_solver object
    T0: initial temperature float
    adaptive: Boolean of whether adaptive cooling is wanted in the implementation
    restarts: Boolean of whether restarts are wanted
    repeats: Number of repeated trials
    
    prints out the results on the terminal
    
    '''

    best_x = None
    best_solution = np.inf    

    solutions = []
    t = 'sigma'
    if T0 == T0_df_average: t = 'df_average'
    d = 'exponential'
    if adaptive : d= 'adaptive'
    r = 'allowed'
    if not restarts: r ='not allowed'


    for i in range(repeats):
        my_solver.initialise()
        my_solver.set_T0(T0)
        my_solver.set_adaptive(adaptive)
        my_solver.set_restarts(restarts)
        res = my_solver.solve()
        x,value ,archive = res['x'], res['fun'], res['archive']
        
    
        solutions.append(value)
        if value < best_solution:
            best_solution = value
            best_x = x
        
        #If issues with colorama install, either 'pip3 install colorama' on terminal or comment out next line
        progress_bar(i+1,repeats)

    print('-'*162)
    print('')
    print(f'Best Solution for T0 = {t}, {d} cooling and restarts {r}')
    print('')

    print(f'x = {best_x}')
    print(f'objective function = {best_solution}')
    print(f'Mean of solutions {np.mean(solutions)}')
    print(f'Standard deviation of solutions {np.std(solutions)}')
    print('')
    print('-'*162)
    print('')



if __name__ == '__main__':

    #Create solver object
    my_solver = SA_Solver(6)

    #Initial survey with T0 = inf and no restarts
    my_solver.set_T0(np.inf)
    my_solver.set_restarts(False)
    res = my_solver.solve()
    x,value ,archive = res['x'], res['fun'], res['archive']

    accepted = archive['accepted']
    observed = archive['observed'].get(np.inf)


    T0_sigma = np.std(observed)
    df_average = np.mean(accepted)
    print(df_average)

    T0_df_average = -df_average/np.log(0.8)
    print(f'Choices for T0 are {T0_sigma} using standard deviation and {T0_df_average} using average acceptance')


    #50 repeats
    repeats = 50

    # T0 = df_average, geometric cooling, restarts allowed
    main(my_solver,T0_df_average,False,True,repeats)
    
    # T0 = df_average, adaptive cooling, restarts allowed
    main(my_solver,T0_df_average,True,True,repeats)

    # T0 = df_average, geometric cooling, restarts not allowed
    main(my_solver,T0_df_average,False,False,repeats)

    # T0 = df_average, adaptive cooling, restarts not allowed
    main(my_solver,T0_df_average,True,False,repeats)


    # T0 = T0_sigma, geometric cooling, restarts allowed
    main(my_solver,T0_sigma,False,True,repeats)
    
    # T0 = T0_sigma, adaptive cooling, restarts allowed
    main(my_solver,T0_sigma,True,True,repeats)

    # T0 = T0_sigma, geometric cooling, restarts not allowed
    main(my_solver,T0_sigma,False,False,repeats)

    # T0 = T0_sigma, adaptive cooling, restarts not allowed
    main(my_solver,T0_sigma,True,False,repeats)

    
