import numpy as np
from pylab import * 

x_values = arange(0.0, math.pi * 4, 0.01)  
y_values = sin(x_values) 
plot(x_values, y_values, linewidth=1.0) 
xlabel('x')  
ylabel('love(x)')  
title('love coding')  
grid(True)
savefig("love_code.png") 
show() 

X = np.arange(-5.0, 5.0, 0.1)
Y = np.arange(-5.0, 5.0, 0.1)

x, y = np.meshgrid(X, Y)
f = 17 * x ** 2 - 16 * np.abs(x) * y + 17 * y ** 2 - 225

fig = figure()
cs = contour(x, y, f, 0, colors = 'r')
savefig("love_code.png") 
show()