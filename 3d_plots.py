import matplotlib.pyplot as plt
import numpy as np
from pylab import meshgrid


'''Run this file to get plots of to give a visualistion of the Schwefel function for 2D 
'''

def objective_function(x,y):

    return -x*np.sin(np.sqrt(abs(x))) -y*np.sin(np.sqrt(abs(y)))

x = np.linspace(-500,500,1000)
y = np.linspace(-500,500,1000)
X,Y = meshgrid(x,y)
Z = objective_function(X,Y)
cmap = 'gist_earth'

fig = plt.figure(figsize=(8, 6))
ax = plt.axes(projection='3d')
ax.plot_surface(X, Y, Z, cmap=cmap)
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('f(x1,x2)')
ax.view_init(60, 35)
plt.show()
con = plt.contourf(x,y,Z, cmap=cmap,extend = 'both',levels = np.linspace(-750,750,51))
plt.colorbar(con)
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()
#plt.savefig('Schhwefel_function')
plt.contour(X, Y, Z, 15, cmap=cmap)
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()

