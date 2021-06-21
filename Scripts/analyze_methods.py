import ast
import pandas as pd

with open("results_methods.txt", "r") as f:
    content = f.read()
methods = ast.literal_eval(content)
methods_count_pd = pd.Series(methods["pandas"])
methods_count_np = pd.Series(methods["numpy"])
methods_count_sklearn = pd.Series(methods["sklearn"])
methods_count_keras = pd.Series(methods["keras"])
methods_count_matplotlib = pd.Series(methods["matplotlib"])

methods_count_pd.value_counts().to_csv("values_count_pd.csv")
methods_count_np.value_counts().to_csv("values_count_np.csv")
methods_count_sklearn.value_counts().to_csv("values_count_sklearn.csv")
methods_count_keras.value_counts().to_csv("values_count_keras.csv")
methods_count_matplotlib.value_counts().to_csv("values_count_matplotlib.csv")