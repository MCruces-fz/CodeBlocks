#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 12 11:57:36 2020

mcsquared.fz@gmail.com

@author: Xoan

Editado:
Miguel Cruces
+34 636 718 231
Call me, BB
"""

import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# Temperaturas T1, T2 y T3
T = np.array([298.15, 350.15, 400.15])

# Fracciones molares
x = np.arange(0.1, 1, 0.1)

# Magnitudes h y v para las temperaturas T1, T2, y T3 respectivamente
h1 = np.array([26.5, 48., 63.7, 73.4, 77.1, 74.6, 65.8, 50.4, 28.3])
v1 = np.array([.13, .25, .4, .55, .69, .77, .78, .68, .46])

h2 = np.array([27.4, 55.5, 80., 99.2, 111.2, 114.1, 106.1, 85.3, 49.7])
v2 = np.array([.47, .78, 1.08, 1.32, 1.48, 1.53, 1.46, 1.22, .79])

h3 = np.array([28.5, 62.8, 95.9, 124.2, 144.1, 152.3, 145.1, 119., 70.6])
v3 = np.array([.63, 1.14, 1.57, 1.89, 2.08, 2.11, 1.95, 1.57, .96])


def func(x: float, *coefs: int):
    """
    Esta función calcula el resultado de las ecuaciones 'h' y 'v' para cada
    valor dado del parámetro 'x' con los coeficientes '*coefs' y los
    devuelve como un array plano contiguo (unidimensional)
    :param x: valor de cada punto x
    :param coefs: valor de los coeficientes A, C, D, E
    :return: array de floats plano contiguo
    """
    h, v = np.array([]), np.array([])
    A, C, D, E = coefs
    for Ti in T:
        h_x = x * (1 - x) * (A + (D + E * (Ti - T[0])) * x + C * x ** 2)
        v_x = x * (1 - x) * (((D + E * (Ti - T[0])) / 100) + C * x)
        h = np.append(h, h_x)
        v = np.append(v, v_x)
    h1t, h2t, h3t = np.split(h, 3)
    v1t, v2t, v3t = np.split(v, 3)
    result = np.array([h1t, h2t, h3t, v1t, v2t, v3t]).T
    return result.ravel()  # numpy ravel convierte en array plano contiguo


ig = [0, 0, 0, 0]
y = np.array([h1, h2, h3, v1, v2, v3]).T
# Tambien es necesario el ravel en xdata e ydata, porsupuesto
p_opt, p_cov = curve_fit(f=func, xdata=x.ravel(), ydata=y.ravel(), p0=ig)

# Valores calculados mediante los coeficientes
result = np.asarray(np.split(func(x, *p_opt), 9))  # (Aquí se deshace el ravel)

"""
============================== RESULTADO ==============================

    Simplemente había que pasarle los datos como un array unidimensional
sin importar el orden de los datos. Lo único que que pasa es que tienen
que concordar los datos de xdata e ydata, por supuesto.

    Una vez conseguido el resultado ya se pueden separar en arrays y 
meterlos todos en uno. Los coeficientes A, C, D y E están dentro de
el output de curve_fit: p_opt

========================================================================
"""

# Parámetros del modelo que describen simultáneamente el comportamiento
# experimental dado por los valores de las magnitudes h y v

A, C, D, E = p_opt

# Valores de B = D + E * (Ti - T0)

B = np.asarray([D + E * (Ti - T[0]) for Ti in T])

#  Representación gráfica

plt.figure(0)
plt.title("Magnitudes h")
y_arr = result.T
plt.plot(x, y_arr[0], 'k', label='T1')
plt.plot(x, y_arr[1], 'r', label='T2')
plt.plot(x, y_arr[2], 'b', label='T3')
plt.xlabel('Fracción molar x')
plt.ylabel('Magnitud h')
plt.legend(loc='best')

plt.figure(1)
plt.title("Magnitudes v")
y_arr = result.T
plt.plot(x, y_arr[3], 'k', label='T1')
plt.plot(x, y_arr[4], 'r', label='T2')
plt.plot(x, y_arr[5], 'b', label='T3')
plt.xlabel('Fracción molar x')
plt.ylabel('Magnitud v')
plt.legend(loc='best')

plt.show()
