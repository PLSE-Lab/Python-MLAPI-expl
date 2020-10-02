import numpy as np 
import torch 
from torch.autograd import Variable
import matplotlib.pyplot as plt

x_dom=np.arange(-1,1,0.01)
ind=np.random.randint(0,len(x_dom),6)
noise = np.random.rand(len(x_dom))
y_dom=-4*(x_dom**2)+3*x_dom+(noise*5)
x=x_dom[ind];y=y_dom[ind]

poly_deg = 9
lrn_rt = 0.1
we_de = 0.0022
epochs = 500
feat=np.ones((len(x),poly_deg+1))


for i in range(poly_deg+1):
	feat[:,i] = x**i
weights = Variable(torch.rand(poly_deg+1),requires_grad=True)
features = Variable(torch.from_numpy(feat),requires_grad=False)
labels = Variable(torch.from_numpy(y),requires_grad=False)
features=features.float()
weights=weights.float()
labels=labels.float()
w_loc=weights

for epoch in range(epochs):
	for i in range(len(x)):
		y_pred = torch.dot(w_loc,features[i,:])
		mse_err = 0.5*((y_pred-labels[i]).pow(2))
		if not weights.grad is None:
			weights.grad.data.zero_()
		mse_err.backward()
		w_loc = w_loc - weights.grad.data*lrn_rt - lrn_rt*we_de*w_loc
		weights.grad.data.zero_()
	if not(epoch%10):
		print(w_loc.data)
weight = w_loc.detach().numpy()
ans = np.polynomial.polynomial.Polynomial(coef=weight)
y_predicted = ans(x_dom)
plt.plot(x_dom,y_predicted,x_dom,y_dom)