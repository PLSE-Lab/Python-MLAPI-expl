import matplotlib.pyplot as plt
from matplotlib.figure import Figure

Figure(figsize=(100,300))

lams = [0.0, 0.0009, 0.001, 0.0025, 0.005, 0.006, 0.0075, 0.01, 0.015, 0.02]
logr_accs = [0.5955, 0.5929, 0.5929, 0.5908, 0.5883, 0.5760, 0.5732, 0.5731, 0.5586, 0.5586]
svm_accs = [0.5890, 0.5827, 0.5827, 0.5815, 0.5822, 0.5731, 0.5731, 0.5731, 0.5586, 0.5586]
rfc_accs = [0.5940, 0.5940, 0.5941, 0.5915, 0.5890, 0.5762, 0.5731, 0.5731,0.5586, 0.5586]
base_x = [0.0, 0.02]
base_y = [0.5049, 0.5049]

plt.plot(lams, logr_accs, marker='.', markersize=10, label='Logistic Regression', alpha=0.3)
plt.plot(lams, svm_accs, marker='.', markersize=10, color='magenta', label='Linear SVM', alpha=0.3 )
plt.plot(lams, rfc_accs, marker='.', markersize=10, color='green', label='Random Forest', alpha=0.3 )
plt.plot(base_x, base_y, color='black', label='baseline')
plt.xlabel('Lambda')
plt.xlim(base_x[0], base_x[1])
plt.ylabel('Cross-Validation Accuracy')
plt.ylim(0.4, 0.7)
plt.axvline(0.0009, linestyle='--', color='red', label='15 predictors') #15 predictors
plt.axvline(0.006, linestyle='--', color='purple', label='5 predictors') #5 predictors
#plt.axvline(0.01, linestyle='--', color='black', label='3 predictors') #3 predictors
plt.axvline(0.015, linestyle='--', color='cyan', label='1 predictor') #1 predictor
plt.xticks(rotation=45)
plt.legend(loc='upper right')
plt.title('Logistic Regression and SVM')
plt.show()