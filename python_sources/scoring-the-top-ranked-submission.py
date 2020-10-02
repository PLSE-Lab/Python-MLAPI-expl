# Here, I show how to read in competition solution, the winning submission, and then compute
# its public score and its private score. The output is shown in the log.

import h5py
import numpy as np

# This won't work correctly in Python 2 or if a submission predicts the same place twice for a checkin
def facebook_fast_map3(solution, submission):
    return np.mean((solution[:,0]==submission[:,0]) +
                   (solution[:,0]==submission[:,1])/2 +
                   (solution[:,0]==submission[:,2])/3)

with h5py.File("../input/top_submissions.h5", "r") as f:
    solution = f["Solution"][:]
    public_locs  = solution[:,2]==0
    private_locs = solution[:,2]==1
    rank1 = f["Rank1"][:]
    public_score  = facebook_fast_map3(solution[public_locs, 1:2], rank1[public_locs,:])
    print("Public Score", public_score)
    private_score = facebook_fast_map3(solution[private_locs, 1:2], rank1[private_locs,:])
    print("Private Score", private_score)
