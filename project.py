# Methodes de balayage pour chercher le minimum

import random
from matplotlib.colors import LightSource
import numpy as np
import matplotlib.pyplot as plt
import math
from matplotlib import cm
from matplotlib.ticker import LinearLocator



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
Xg, Yg = np.meshgrid(np.linspace(-10,10), np.linspace(-10,10))
Zg = g(Xg,Yg, 2,2/7)
levelsG = np.linspace(Zg.min(), Zg.max(), 7)

# plot
figg, axg = plt.subplots()

axg.contourf(Xg, Yg, Zg, levels=levelsG)

plt.show()
#H
X, Y= np.meshgrid(np.linspace(-10,10), np.linspace(-10,10))

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
    a = 2
    b = 2/7
    index = 0
    points = []
    (xnp1, ynp1) = np.array((x0, y0))
    points.append((xnp1, ynp1))
    while(eps < np.linalg.norm((df1(xnp1, ynp1, a, b), df2(xnp1, ynp1, a, b))) and index < MaxIter):
        (xnp1, ynp1) = (xnp1, ynp1) + u * np.array((df1(xnp1, ynp1, a, b), df2(xnp1, ynp1, a, b)))
        points.append((xnp1, ynp1))
    return points


def gradpc2(eps, MaxIter, u, x0, y0, df1, df2):
    index = 0
    points = []
    (xnp1, ynp1) = np.array((x0, y0))
    points.append((xnp1, ynp1))
    while(eps < np.linalg.norm((df1(xnp1, ynp1), df2(xnp1, ynp1))) and index < MaxIter):
        (xnp1, ynp1) = (xnp1, ynp1) + u * np.array((df1(xnp1, ynp1), df2(xnp1, ynp1)))
        points.append((xnp1, ynp1))
    return points

#C.13#---------------
a = 2
b = 2/7

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
pointsh = gradpc2(10**-5, 120, -0.1, 0, 0, dphx, dphy)

#G
xs = [x[0] for x in pointsg]
ys = [y[1] for y in pointsg]
"""
Xs, Ys = np.meshgrid(xs, ys)
Zs = g(Xs,Ys,a,b)

Ls = np.linspace(Zs.min(), Zs.max(), 7)
"""
Xg, Yg = np.meshgrid(np.linspace(0,7), np.linspace(-5,5))
Zg = g(Xg,Yg, 2,2/7)
levelsG = np.linspace(Zg.min(), Zg.max(), 15)
figg, axg = plt.subplots()
axg.contourf(Xg, Yg, Zg, levels=levelsG)
axg.plot(xs, ys)
plt.plot()
plt.show()

#H

xs = [x[0] for x in pointsh]
ys = [y[1] for y in pointsh]
X, Y= np.meshgrid(np.linspace(-1,1), np.linspace(-2.5,0))
Z = h(X,Y)
levels = np.linspace(Z.min(), Z.max(), 7)

fig, ax = plt.subplots()

ax.contourf(X, Y, Z, levels=levels)
ax.plot(xs, ys)
plt.plot()
plt.show()


#C.14#-------------------

def gradG(eps, MaxIter, u, x0, y0, df1, df2):
    a = 1
    b = 20
    index = 0
    (xnp1, ynp1) = (x0, y0)
    while(eps < np.linalg.norm((df1(xnp1, ynp1, a, b), df2(xnp1, ynp1, a, b))) and index < MaxIter):
        (xnp1, ynp1) = (xnp1, ynp1) + u * np.array((df1(xnp1, ynp1, a, b), df2(xnp1, ynp1, a, b)))
    return (xnp1, ynp1)



a = 1
b = 20
eps = 10**(-5)
maxiter = 120
u = np.linspace(-0.99, -0.001, 50)
np.linalg.norm((0,0))
valeur_reel = (0, 0)
pointsg = [np.linalg.norm(gradG(10**-5, 120, i, 1, 1, dpgx, dpgy)) for i in u]
print(pointsg)
plt.plot(u,pointsg)
plt.show()