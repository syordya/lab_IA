# -*- coding: utf-8 -*-
#Alumno: SANTOS/APAZA, YORDY WILLIAMS
#Repositorio: https://github.com/syordya/lab_IA

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

X = np.array ([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,
42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80]).reshape(-1,1)

Y = np.array ([31,28,89,29,55,45,32,21,64,100,55,36,181,98,115,258,91,181,151,535,280,393,1388,914,641,951,671,2265,519,1172,
1016,998,931,1208,697,1512,1413,1664,734,3683,2186,1182,2491,2741,3045,3483,2075,3394,1444,3817,3628,3709,3321,3168,2292,1515,
3237,4247,4298,3891,4046,3732,2660,4550,4537,4749,2929,4056,4205,4020,5772,6154,5874,6506,7386,8805,5563,4845,4030,4284]).reshape(-1,1)

reg = LinearRegression().fit(X,Y)

yp = X*reg.coef_+reg.intercept_

plt.plot(X,Y,'.',label="Datos")
plt.plot(X,yp,label="Regresion Lineal")
plt.legend()
plt.show()
