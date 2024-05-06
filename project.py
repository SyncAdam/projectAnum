# Methodes de balayage pour chercher le minimum

import random
from matplotlib.colors import LightSource
import numpy as np
import matplotlib.pyplot as plt
import math
from matplotlib import cm
from matplotlib.ticker import LinearLocator
#B.1# -----------------

def f(x):
    return pow(x, 3) - 3 * pow(x, 2) + 2 * x + 5

def mf(x):
    return -(pow(x, 3) - 3 * pow(x, 2) + 2 * x + 5)

def chercheMinPasConstant(f, a, b, n):
    dx = abs(b - a) / n
    min = f(a)
    for i in range(n):
        if(f(a + dx * i) < min):
            min = f(a + dx * i)
    return min

def BalAlea(f,a,b,N):
    Val = f(a+ (b-a)*random.random())
    for i in range(N+1):
        N = f(a+(b-a)*random.random())
        if(Val>N):
            Val=N
    return Val

#B.2# -----------------

print(chercheMinPasConstant(f,0,3,1000))
print(BalAlea(f,0,3,1000))    

#B.3# ------------------
#À calculer à la main 
minreelle = (3 + np.sqrt(3)) / 6
n = 1000
a = 0
b = 3

pasConstantErreur = []
aleatoireErreur= []

for i in range(n):
    pasConstantErreur.append(abs(minreelle - chercheMinPasConstant(f, a, b, i + 1)))
    aleatoireErreur.append(abs(minreelle - BalAlea(f, a, b, i + 1)))

X = [k + 1 for k in range(n)]

#plt.plot(np.log(X), np.log(pasConstantErreur), 'r-', label="Erreur pour la methode pas constant")
#plt.plot(np.log(X), np.log(aleatoireErreur), 'b-', label="Erreur pour la methode pas aleatoire")
#plt.ylabel('ln(err)')
#plt.xlabel ('X')
#plt.legend()
#plt.show()


#B.4# -----------------

# pour chercher le maximum de f avec la meme methode
# mf représente -f
print(-chercheMinPasConstant(mf, a, b, n))
print(-BalAlea(mf, a, b, n))

#B.6# -----------------
# méthodes du gradiant 1D

x0 = a + (b - a) * random.random()

u = -0.01  #t = -0.28876 pour optimale

def fder(x):
    return (3 * math.pow(x, 2) - 6 * x + 2)

for i in range(n):
    if(i == 0):
        xnp1 = x0 + u * fder(x0)
    else:
        xnp1 = xnp1 + u * fder(xnp1)
        #print(xnp1)

print(f(xnp1))
print("xnp1" , xnp1)




#B.7# -----------------
    #B.7.a# -----------------

def fder2(x): #f''(x)
    return 6 * x - 6

def phi(t): #𝜑(t)
    return f(xnp1 + t * fder(xnp1))

def phider(t):#𝜑'(t)
    return (fder(xnp1 + t * fder(xnp1)) * fder(xnp1))

def phider2(t): #𝜑''(t)
    return (fder2(xnp1 + t * fder(xnp1)) * math.pow(fder(xnp1), 2))

print(phider(0))
print(phider2(0))

    #B.7.b# -----------------

def phiDL(t): 
    return phider(0) + phider2(0) * t

#t = (-phider(0)/phider2(0))


#C.8# -----------------

#  Nabla <=> derivée partiel
# plot_surface 

def g(x, y, a, b):
    return (pow(x, 2) / a) + pow(y, 2) / b

def h(x, y):
    A = np.cos(x)
    B = np.sin(y)
    return (A)*(B)

x = np.arange(-10, 10, 0.25)
y = np.arange(-10, 10)

a = 2
b = 2/7

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

x, y = np.meshgrid(x, y)

z =  g(x, y, a, b)

# Customize the z axis.
ax.plot_surface(x, y, z, vmin=z.min() * 2, cmap=cm.Blues)

ax.set(xticklabels=[],
       yticklabels=[],
       zticklabels=[])

plt.show()

x1 = np.arange(-10, 10, 0.1)
y1 = np.arange(-10, 10, 0.1)
x1, y1 = np.meshgrid(x1, y1)
z1 = h(x1, y1)

fig1, ax1 = plt.subplots(subplot_kw={"projection": "3d"})
ax1.plot_surface(x1, y1, z1, vmin=z1.min() * 2, cmap=cm.Blues)
ax1.set(xticklabels=[],
       yticklabels=[],
       zticklabels=[])
plt.show()


#C.9#----------------

plt.style.use('_mpl-gallery-nogrid')
#G
# make data
X, Y = np.meshgrid(np.linspace(-10,10), np.linspace(-10,10))
Z = g(X,Y, 2,2/7)
levels = np.linspace(Z.min(), Z.max(), 7)

# plot
fig, ax = plt.subplots()

ax.contourf(X, Y, Z, levels=levels)

plt.show()
#H
Z = h(X,Y)
levels = np.linspace(Z.min(), Z.max(), 7)

# plot
fig, ax = plt.subplots()

ax.contourf(X, Y, Z, levels=levels)

plt.show()

#C.10#---------------

def gradg(x, y, a, b):
    return(((2*x)/a ) + (y**2)/b, (x**2)/a + (2*y)/b)

def gradh(x, y):
    return (-np.sin(x)*np.sin(y), np.cos(x)* np.cos(y))

#C.11#---------------
point = (0, 1)
point2 = (1, 1)

print(gradg(point[0], point[1], a, b))
print(gradh(point[0], point[1]))
print(np.linalg.norm(gradg(point[0], point[1], a, b)))
#C.12#---------------

#pas constant



def gradpc(eps, MaxIter, u, x0, y0, df1, df2):
    index = 0
    points = []
    (xnp1, ynp1) = (x0, y0) 
    points.append((xnp1, ynp1))
    while(eps < np.linalg.norm((df1, df2)) and index < MaxIter):
        (xnp1, ynp1) = (xnp1, ynp1) + u*(df1(xnp1, ynp1), df2(xnp1, ynp1))
        points.append((xnp1, ynp1))
    return points

#C.13#---------------

(x01, y01) = (0, 0)

#derivée partielle par rapport a x de g
def dpgx(x, y, a, b):
    return (((2*x)/a ) + (y**2)/b)

#derivée partielle par rapport a y de g
def dpgy(x, y, a, b):
    return (x**2)/a + (2*y)/b

def dphx(x, y):
    return -np.sin(x)*np.sin(y)

def dphy(x, y):
    return np.cos(x)* np.cos(y)

pointsg = gradpc(10**-5, 120, -0.1, 7, 1.5, dpgx, dpgy)
pointsh = gradpc(10**-5, 120, -0.1, 0, 0, dphx, dphy)







