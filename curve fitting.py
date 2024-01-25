import numpy as np
from scipy.optimize import curve_fit
import math
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('D:/luxathon/DATA SET/data.csv')
x_data = data['LDR Resistance']
y_data = data['Illuminance']

def curve_func(x, a, b, c, d, e, f, g, h,i, j): return a*x**4+b*x**3+c*x**2+i*x**0


params,covariance = curve_fit(curve_func, x_data,y_data)


a_fit, b_fit, c_fit, d_fit, e_fit,f_fit,g_fit,h_fit,i_fit,j_fit = params


x_fit = np.linspace(10,1000,65)
y_fit = curve_func(x_fit, a_fit, b_fit, c_fit, d_fit, e_fit,f_fit,g_fit,h_fit,i_fit,j_fit)


plt.scatter(x_data, y_data, label='Original Data')
plt.plot(x_fit, y_fit,'r', label='Fitted Curve')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Curve Fitting')
plt.legend()
plt.grid(True)
plt.show()


print('Coeffiecients: {}', format([x_fit, a_fit, b_fit, c_fit, d_fit, e_fit,f_fit,g_fit,h_fit,i_fit,j_fit]))
