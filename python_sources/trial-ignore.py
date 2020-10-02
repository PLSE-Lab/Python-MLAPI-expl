x = [0] * 5 + [1] * 5
print(x)

tpr_thresholds=(0.2, 0.4, 0.6, 0.8)
roc_weights=(4, 3, 2, 1, 0)
tpr_thresholds = [0.] + list(tpr_thresholds) + [1.]
print(tpr_thresholds)