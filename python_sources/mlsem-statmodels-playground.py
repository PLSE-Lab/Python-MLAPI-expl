from __future__ import print_function
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.sandbox.regression.predstd import wls_prediction_std

def statmodels_playground():
    nSamples = 100

    x1 = np.linspace(0, 10, 100)
    x2 = np.linspace(-3, 6, 100)
    dataMatrix = np.column_stack((x1, x2))

    coefficients = np.array([1, 0.1, 20])

    errorTerm = np.random.normal(size=nSamples, scale=5)
    dataMatrix = sm.add_constant(dataMatrix)
    outputValues = np.dot(dataMatrix, coefficients) + errorTerm
    model = sm.OLS(outputValues, dataMatrix)
    results = model.fit()
    print(results.summary())
    print('Parameters: ', results.params)
    print('R2: ', results.rsquared)
    
statmodels_playground()