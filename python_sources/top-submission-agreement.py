import h5py
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set(style="white")

num_top = 10
 
def submission_ranked_aggreement(sub1, sub2):
    max_score = 0.0
    scores = 0.0*sub1[:,0]
    for i in range(sub1.shape[1]):
        max_score += 1/(i+1)/2
        for j in range(i+1):
            scores += (sub1[:,i]==sub2[:,j])/(i+j+2)
    return np.mean(scores)/max_score
    
def submission_top_1_aggreement(sub1, sub2):
    assert sub1.shape==sub2.shape
    return np.mean(sub1[:,0]==sub2[:,0])

def plot_agreement(pairwise_agreement, filename):
    f, ax = plt.subplots(figsize=(11, 9))
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    sns.heatmap(pairwise_agreement, cmap=cmap, vmax=1, vmin=0,
                square=True, linewidths=.5, cbar_kws={"shrink": 0.8}, ax=ax,
                annot=True, fmt="0.2f", 
                xticklabels=range(1, pairwise_agreement.shape[0]+1),
                yticklabels=range(1, pairwise_agreement.shape[1]+1))
    plt.savefig(filename)
    plt.close()

with h5py.File("../input/top_submissions.h5", "r") as f:
    solution = f["Solution"][:]
    private_locs = solution[:,2]==1
    pairwise_top_1_agreement  = np.zeros((num_top,num_top))+np.nan
    pairwise_ranked_agreement = np.zeros((num_top,num_top))+np.nan
    for row in range(num_top):
        row_submission = f["Rank%d" % (row+1)][:]
        for col in range(row+1):
            col_submission = f["Rank%d" % (col+1)][:]
            pairwise_top_1_agreement[row, col]  = submission_top_1_aggreement(row_submission[private_locs,:], col_submission[private_locs,:])
            pairwise_ranked_agreement[row, col] = submission_ranked_aggreement(row_submission[private_locs,:], col_submission[private_locs,:])
    plot_agreement(pairwise_top_1_agreement, "top_1_agreement.png")
    plot_agreement(pairwise_ranked_agreement, "ranked_agreement.png")
