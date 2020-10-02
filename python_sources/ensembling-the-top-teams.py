from collections import Counter
import h5py
import matplotlib.pyplot as plt
from ml_metrics import mapk
import numpy as np
import pandas as pd
import seaborn as sns
sns.set(style="white")
 
num_top=10

def facebook_v_ensembler(submission, state):
    if state is None:
        state = [Counter() for i in range(submission.shape[0])]
    for i in range(submission.shape[0]):
        for j in range(submission.shape[1]):
            state[i][submission[i,j]] += (3-j)
    preds = [[x[0] for x in s.most_common(3)] for s in state]
    return np.array(preds), state

# This won't work correctly if a submission predicts the same place twice for a checkin
def facebook_fast_map3(solution, submission):
    return np.mean((solution[:,0]==submission[:,0]) +
                   (solution[:,0]==submission[:,1])/2 +
                   (solution[:,0]==submission[:,2])/3)

with h5py.File("../input/top_submissions.h5", "r") as f:
    solution = f["Solution"][:]
    private_locs = solution[:,2]==1
    
    scores = []
    state = None
    for i in range(1, num_top+1):
        submission = f["Rank%d" % i][:]
        team_score = facebook_fast_map3(solution[private_locs, 1:2], submission[private_locs,:])
        ensemble, state = facebook_v_ensembler(submission[private_locs,:], state)
        ensemble_score = facebook_fast_map3(solution[private_locs, 1:2], ensemble)
        print("%d. Team Score %0.4f, Ensemble Score %0.4f" % (i, team_score, ensemble_score))
        scores.append([i, team_score, ensemble_score])

data = pd.melt(pd.DataFrame(scores, columns=["Place", "Single Team", "Ensemble"]),
               ["Place"], ["Single Team", "Ensemble"], "Type", "Score")
sns.factorplot(x="Place", y="Score", hue="Type", data=data, size=6, aspect=1.5)
plt.savefig("ensemble.png", dpi=600)
