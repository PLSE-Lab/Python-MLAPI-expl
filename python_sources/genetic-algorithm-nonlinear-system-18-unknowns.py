#!/usr/bin/env python
# coding: utf-8

# # Genetic Algorithm (GA): A Quantitative Approach to a System of 16 Non-Linear Equations with 18 Unknowns

# # Introduction
# 
# Greetings, at the time I am writing this, I just turned 17 and have just started my first year Junior College just 2 months before COVID-19 confined us to video conference home-based school lessions at home. 
# 
# Out of nowhere during COVID-19 lock, my Physics teacher dropped us thie following maths challenge and challenged us to come up with some programme code to find the solution(s) of the following non-linear system of 16 equations with 18 unknowns.
# 
# He does not have the solution(s) nor does he know if there is a solution at all.
# 
# This notebook documents my approach to the challenge using Genetic Algorithm.
# 

# # Problem Statement
# 
# D1, D2, D3, D4, D5, D6, D7, D8, D9 and D10 are all single digit positive integers, i.e they are 0, 1, 2...or 9. Digits may repeat, and not all digits need to be used.
# 
# Q to X are also single digit positive integers between 0 - 9, just like D1 - D10.
# 
# The mission is to find the solutions (D1 to D10 and Q to X) that satisfy all the equations.
# 
# 1. Q = D2 + D4
# 1. R = D5 + D2 + 1
# 1. S = D1 - D3 + D2
# 1. T = D4 + D5 + 1
# 1. U = D1 - D5
# 1. V = D3 - D5 + D2
# 1. W = D4 + 1
# 1. X = D3 + D5
# 
# 1. Q = D10 - D9 - D8
# 1. R = D8 + D9
# 1. S = D8 x 2
# 1. T = D6 + D9
# 1. U = D10 - D8 - D7
# 1. V = (D7 x D9) + D7
# 1. W = D7 + D6
# 1. X = (D8 x D9) + D7
# 

# # My Approach
# ## Observation
# 
# Breaking down the question, the equations are non-linear and therefore, cannot be solved using linear matrix algebra. There are 18 unknowns in 16 equations, so the system cannot be solved uniquely as it will result in a collection of solutions with 2 degrees of freedom due to the additional 2 unknowns than equations. I have tried to approach it analytically but to no avail, so I have decided to approach it quantitatively.
# 
# ## Quantitative Approach
# 
# Since each of the 18 variables takes discrete values in between integers 0 and 9, this solution universe has 10$^{18}$ possibilities. If a solution does exist, it will be among this universe of 18 possibilities. If I look at it as a search for the solution among 10$^{18}$ possiblities, it becomes a search problem rather than a math problem. 
# 
# The question becomes: if the solution exists, what is the most effective way to search among this space to locate that solution or a set of solutions. 
# 
# The most simplistic approach is to do a grid search for 10$^{18}$, but this is inefficient and uses brute force. Upon further observation, the variables {D1-D10, Q-X} looks like a gene with 18 strings of DNA which can take 1 of 10 integers (0 - 9). If I can encode the orginal question into a genetic structure, I will be able to deploy Genetic Algorithm as my search engine.
# 
# This notebook documents my approach to the challenge using Genetic Algorithm.

# # Genetic Algorithm Approach
# 
# The Genetic Algorithm is inspired from Darwin's theory of natural evolution. This is reflected in the algorithm where it selects the fittest individuals from each "generation" to reproduce offspring for the next generation, making each generation better than the last. Each individual can be categorised by their genes (variables) which come together to form a chromosome (solution). The population consists of multiple indivduals with varying chromosomes with varying combinations of genes. Each generation of the population is then put through a "test" and give a fitness score, after which, the top percentage of the population (which can be set by the user) is selected to enter the reproductive stage. This top percentage is decided by how close each individual's fitness score is to the desired fitness score. During the reproductive stage, the individuals "mate" to form offsprings with certain components of their genes. These offsprings form the next generation and the cycle will continue.
# 

# # Formulation of Problem into Genetic Algorithm
# ## Coding the problem into genetic code
# 
# DNA is the molecule that is the hereditary material in all living cells. Genes are made of DNA, and so is the genome itself. A gene consists of enough DNA to code for one protein, and a genome is simply the sum total of an organism's DNA. The information in DNA is stored as a code made up of four chemical bases: adenine (A), guanine (G), cytosine (C), and thymine (T).
# 
# In our problem, we encode our problem as a gene structure with 18 "DNA"s. Each such "DNA" takes values of of 10 values {0,1,2,3,4,5,6,7,8,9}.
# 
# For example, encoded solution is a gene of this structure {D1,D2,D3,D4,D5,D6,D7,D8,D9,D10,Q,R,S,T,U,V,W,X}
# 
# ## Genetic Superiority Function (Gene Score)
# 
# To allow genetic evoluation based survival of the best, we need define what is considered a good gene. I define gene score function to evaluate superiority of a gene.
# 
# I rearrange the orginal 16 equations to put all variable on the right hand side so the right hand side is 0. I define the gene score function as follow:
# 
# 1. score_1 = D2 + D4 - Q
# 1. score_2 = D5 + D2 + 1 - R
# 1. score_3 = D1 - D3 + D2 - S
# 1. score_4 = D4 + D5 + 1 - T
# 1. score_5 = D1 - D5 - U
# 1. score_6 = D3 - D5 + D2 - V
# 1. score_7 = D4 + 1 - W
# 1. score_8 = D3 + D5 - X
# 1. score_9 = D10 - D9 - D8 - Q
# 1. score_10 = D8 + D9 - R
# 1. score_11 = D8 * 2 - S
# 1. score_12 = D6 + D9 - T
# 1. score_13 = D10 - D8 - D7 - U
# 1. score_14 = (D7 * D9) + D7 - V
# 1. score_15 = D7 + D6 - W
# 1. score_16 = (D8 * D9) + D7 - X
# 
# I define the gene score as:
# 
# gene_score = abs(score_1) + abs(score_2) + abs(score_3) + abs(score_4) + abs(score_5) + abs(score_6) + abs(score_7) + abs(score_8) + abs(score_9) + abs(score_10) + abs(score_11) + abs(score_12) + abs(score_13) + abs(score_14) + abs(score_15) + abs(score_16)
# 
# The lowest the gene_score, the more superior the gene is. A solution is found when the gene has a gene_score of 0 (i.e. the gene can satisfy all the equations to be 0)
# 

# In[ ]:


def gene_score(arr_gene):    
    D1 = arr_gene[:,1]
    D2 = arr_gene[:,2]
    D3 = arr_gene[:,3]
    D4 = arr_gene[:,4]
    D5 = arr_gene[:,5]
    D6 = arr_gene[:,6]
    D7 = arr_gene[:,7]
    D8 = arr_gene[:,8]
    D9 = arr_gene[:,9]
    D10 = arr_gene[:,10]
    Q = arr_gene[:,11]
    R = arr_gene[:,12]
    S = arr_gene[:,13]
    T = arr_gene[:,14]
    U = arr_gene[:,15]
    V = arr_gene[:,16]
    W = arr_gene[:,17]
    X = arr_gene[:,18]
    
    score_1 = D2 + D4 - Q
    score_2 = D5 + D2 + 1 - R
    score_3 = D1 - D3 + D2 - S
    score_4 = D4 + D5 + 1 - T
    score_5 = D1 - D5 - U
    score_6 = D3 - D5 + D2 - V
    score_7 = D4 + 1 - W
    score_8 = D3 + D5 - X
    score_9 = D10 - D9 - D8 - Q
    score_10 = D8 + D9 - R
    score_11 = D8 * 2 - S
    score_12 = D6 + D9 - T
    score_13 = D10 - D8 - D7 - U
    score_14 = (D7 * D9) + D7 - V
    score_15 = D7 + D6 - W
    score_16 = (D8 * D9) + D7 - X
        
    score = np.absolute(score_1)     + np.absolute(score_2)     + np.absolute(score_3)     + np.absolute(score_4)     + np.absolute(score_5)     + np.absolute(score_6)     + np.absolute(score_7)     + np.absolute(score_8)     + np.absolute(score_9)     + np.absolute(score_10)     + np.absolute(score_11)     + np.absolute(score_12)     + np.absolute(score_13)     + np.absolute(score_14)     + np.absolute(score_15)     + np.absolute(score_16)

    return score


# ## Genetic Evolution: Survival of the Best
# 
# Initially, we created a random population of individuals (N_POP). The genetic codes are randomly created. After that, we calculated the genetic score for each gene in the population which calculates a certain percentage of the best genes and discards the remaining genes. It then randomly mates the superior genes to create a new generation and fills up the gaps left by the discarded population with the offspring of the previous best. It also swaps the DNA of the children to let them inherit parts of their parents in hopes of achieving a lower score. Some of the offspring may randomly inherit the good DNA's to perform better their parents while some may inherit inferior DNA's by chance that cause them to perform worse than their parents. After running this for n number of revolutions to see the evolution, we create the best genetic makeup. Hopefully it reaches to 0 to obtain the best solution.

# In[ ]:


def create_init_generation(n_pop):
    arr_gene = np.random.randint(0, high=10, size=(n_pop, 19))
    arr_gene[:,0] = -1
    return arr_gene

def update_gene_score(arr_gene, gene_score):
    # ----- select best gene by score -----
    arr_gene[:,0] = gene_score(arr_gene)
    
    return arr_gene

def mate_best_gen(arr_gene, n_pop_tar, mate_best_pct, gene_score):
    arr_gene = np.unique(arr_gene, axis=0)
    # get population of best genes for mating to restore full population
    n_pop = arr_gene.shape[0]
    n_best_gene = int(n_pop_tar * mate_best_pct)
    if n_pop < n_best_gene:
        n_best_gene = n_pop
    n_loop = n_pop_tar // n_best_gene
    # ----- select best gene by score -----
    arr_gene = update_gene_score(arr_gene, gene_score)
    arr_gene = arr_gene[np.argsort(arr_gene[:, 0])]
    arr_best = arr_gene[:n_best_gene][:]
    # ----- cross mate best genes to generate whole population -----
    arr_pop = arr_best
    is_ans = False
    for i in range(n_loop):
        arr_mask1 = np.random.randint(0, high=2, size=arr_best.shape)
        arr_mask2 = 1 - arr_mask1
        arr_best_shuffle = arr_best.copy()
        np.random.shuffle(arr_best_shuffle)
        arr_tmp = arr_best * arr_mask1 + arr_best_shuffle * arr_mask2
        arr_tmp = update_gene_score(arr_tmp, gene_score)
        arr_pop = np.vstack((arr_pop, arr_tmp))
        #display('loop {}: best score = {}'.format(i+1, arr_tmp[0][0]))    
    arr_pop = np.vstack((arr_pop, create_init_generation(n_best_gene)))
    # ----- select best gene by score -----
    arr_gene = np.unique(arr_pop, axis=0)
    arr_gene = update_gene_score(arr_gene, gene_score)
    arr_gene = arr_gene[np.argsort(arr_gene[:, 0])]
    arr_gene = arr_gene[:n_pop_tar,:]
    best_ans = arr_gene[0][0]

    '''
    display('Best Answer: {}'.format(best_ans))
    display(arr_gene)
    '''
    if best_ans == 0:
        is_ans = True
    
    return is_ans, arr_gene


# ## Problem of local minimum against global minimum
# 
# My goal is to acheive an efficient multidimensional search in an n dimensional space. In this case, there are 18 dimensions as there are 18 variables. Our function, gene_score, defines a surface in a 18-Dimensional space. The genetic algorithm moves along this surface, sensing the gradient of the surface. Depending on luck and the starting point, the generation of the population may end up getting stuck in a bump or groove in the surface which is the local minimum but not the global minimum. To address this issue, I have considered 2 approaches: Genetic Mutation and Random Genetic Settlements.
# 
# ### Genetic Mutation
# 
# Methodically, it will randomly swap genes which do not follow the parents. During evolution, the algorithm will introduce a mutation where genes, that do not come from the best parents, create anomalies that may venture outside of the groove, creating an opportunity that some genes will travel outside the groove.
# 
# To counter this, in addition to the best gene from previous generation (the parents as the benchmark for children to beat) and the cross-bred offsprings from the best genes from previous generation, we add in impurity of randomly generated genes (not from parents) for mutation to give the population to get out of the undesirable in-bred syndome that will get them stuck in a groove.
# 
# ### Diverse Settlements creations
# 
# This solution is to create a few settlements with random genes to explore different local minimums. Each settlement searches for their own minimums. After this, we compare the minimums of different settlements to find the lowest minimum which is the closest solution. 

# In[ ]:


import numpy as np

# number of settlements
N_SETTLEMENT = 50
# number of generations for evolution for each settlement
N_GEN = 20
# size of population for each settlement
N_POP = 100000
# top percent of the best genes for mating to re-populate the next generations
MATE_BEST_PCT = 0.20
# number of top best genes to be retain and stored for each settlement
N_SETTLEMENT_BEST = 50


# In[ ]:


is_ans = False
arr_gene = create_init_generation(n_pop = N_POP)
arr_gene_best = np.zeros((0, arr_gene.shape[1]))
for i_settle in range(N_SETTLEMENT):
    arr_gene = create_init_generation(n_pop = N_POP)
    for i_gen in range(N_GEN):
        is_ans, arr_gene = mate_best_gen(
            arr_gene = arr_gene,
            n_pop_tar = N_POP, 
            mate_best_pct = MATE_BEST_PCT,
            gene_score = gene_score
        )
        best_score = arr_gene[0][0]
        #display(arr_gene[0,:])
        if is_ans:
            print('answer found')
            break
        #display(arr_gene.shape)
        display('settle[{}/{}]-gen[{}/{}]; best score = {}'.format(i_settle + 1, N_SETTLEMENT, i_gen + 1, N_GEN, best_score))
    arr_gene_best = np.vstack((arr_gene_best, arr_gene[:N_SETTLEMENT_BEST,:]))
    shape_b = arr_gene_best.shape
    arr_gene_best = np.unique(arr_gene_best, axis=0)
    shape_a = arr_gene_best.shape
    arr_gene_best = arr_gene[np.argsort(arr_gene_best[:, 0])]
    best_score = arr_gene_best[0][0]
    display('settle[{}][shape.before = {}, shape.after = {}]; best score = {}'.format(i_settle + 1, shape_b, shape_a, best_score))
    np.savetxt('output.csv', arr_gene_best, delimiter=',')
    if is_ans:
        break


# # Results
# 
# I have run the Genetic Evolution for the following parameters:

# In[ ]:


print('Number of Settlements: {}\nPopulation Size: {}\nNumber of Evolutions per Settlement: {}\nTop Best Genes for Mating per Evolution: {}%'.format(N_SETTLEMENT, N_POP, N_GEN, MATE_BEST_PCT * 100))


# With such parameters,the best solution the Genetic Algorithm can find can satisfy 15 equaltions except for 1.
# 
# The following lists the gene code for such best solutions. 

# In[ ]:


import pandas as pd

dict_col_rename = {
    0: 'gene_score',
    1: 'D1',
    2: 'D2',
    3: 'D3',
    4: 'D4',
    5: 'D5',
    6: 'D6',
    7: 'D7',
    8: 'D8',
    9: 'D9',
    10: 'D10',
    11: 'Q',
    12: 'R',
    13: 'S',
    14: 'T',
    15: 'U',
    16: 'V',
    17: 'W',
    18: 'X',    
}
df_result = pd.DataFrame(arr_gene_best)
df_result = df_result.rename(columns=dict_col_rename)

gene_score_best = min(df_result.gene_score)
display('Best Gene Score: {}'.format(gene_score_best))

#display(df_result.query('gene_score == @gene_score_best'))
print(df_result.query('gene_score == @gene_score_best').to_string(index=False))


# # Conclusion
# 
# The main objective of this notebook is to demonstrate the Genetic Algorithm approach to find solution(s) for the given system of non-linear equations in 18 unknowns.
# 
# Genetic Algorithm is a generic search algorithm with applications not only confined to the given systems of non-linear equation. The key is to encode your problem statement into genetic codes which are to be processed by the Genetic Algorithm.
# 
# As for the solutions to the non-linear equations. The best Genetic Algorithm can find can only satisfy 15 out of 16 equations simulataneously. It does not necessarily mean definitely there is no solution; it just means our algorithm with the parameters for the search space could not find one. We can try to extend the search parameters or other improvement.
# 
# Any better solution to find the solution(s) or to prove that solution does not exist is welcome. Please share your approach and I will be thrilled to learn from you.
# 
# Thank you.

# In[ ]:




