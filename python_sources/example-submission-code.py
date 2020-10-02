# This skeleton script provides an example of how subissions must be structured in order to win the LOGOS Speech Recognition Challenge.
#
# Submissions must:
# - take as input a data directory (ie a path string) containing corresponding pairs of
#       - audio sound files (mp3 or wav audio)
#       - target word lists (unformatted text file)
# + and output a comma delimited (.csv) file indicating the presence or absense of words from the target list on their corresponding recordings
#       + these presences or absences should be indicated by 1s or 0s in a vector, one per line corresponding to each recording
#       + the vector must be preceded by a 'case' ID, which is the ID number of the recording/target list as labelled by their respective files
#       + the order of indicator variables (1s and 0s) must match the word order of the target list
#       + the file must have column headers as follows: "Id, R1, R2, ..., R15"

import os
import re
import sys
import glob
from datetime import datetime

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


    
def main(data_dir='../input', solution_file='TeamSIH.csv'):
    print('This is {}, running at {}.\n'.format(sys.argv[0], datetime.now().strftime('%Y-%m-%d %H:%M:%S')))


    # Input data
    print('Using input data directory: {}'.format(data_dir))
    audio_files = glob.glob1(data_dir,"rID*.wav") + glob.glob1(data_dir,"rID*.mp3")
    Nfiles = len(audio_files)

    print('..directory contains {} test audio files:'.format(Nfiles))
    print(os.listdir(data_dir))


    # Generate solution
    # ** This is a random guess solution! **
    # ** Replace this with YOUR ACTUAL solution.. **
    col_names = ['R1','R2','R3','R4','R5','R6','R7','R8','R9','R10','R11','R12','R13','R14','R15']

    # > Get test case labels
    caseIDs = [re.match(r"[rt]ID-(\d+)",file).group(1) for file in audio_files]

    # > Generate random guesses
    df = pd.DataFrame(np.random.randint(low=0, high=2, size=(Nfiles,15)), columns=col_names)
#     df.index += 1
    df['Id'] = pd.to_numeric(caseIDs)
    df=df[['Id']+col_names]

    df.sort_values('Id', inplace=True)
    print()
    print('Solution table:')
    print(df)

    print()
    print('Writing output solution file to: {}'.format(solution_file))
    df.to_csv(solution_file,index=False)

    print('Current directory contains:')
    print(os.listdir('./'))
    
    
if __name__ == '__main__': main()














