from numpy import *
import matplotlib.pyplot as plt

def doPCA(data):
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    pca.fit(data)
    return pca
    
data = genfromtxt("../input/data.csv", delimiter=",") # data with 2 dimensions/features
    
pca = doPCA(data)
print(pca.explained_variance_ratio_) # higher the ratio higher is the information gain
first_pc = pca.components_[0] # best PCA since the information loss is minimum (this PCA is perfect 1D representation of the data)
second_pc = pca.components_[1] # Not the best PCA since the information loss is large

transformed_data = pca.transform(data)
for ii,jj in zip(transformed_data, data):
    #(Transformed_data_column_n)(n_pc_column_1_2)=(co-ordinate_Axis of nth principle component)
    plt.scatter(first_pc[0]*ii[0], first_pc[1]*ii[0], color='r')  # n = first
    plt.scatter(second_pc[0]*ii[1], second_pc[1]*ii[1],color='c') # n = second
    plt.scatter(jj[0], jj[1], color='b')
    
plt.xlabel("bonus")
plt.ylabel("long-term incentive")
plt.savefig('graph.png')