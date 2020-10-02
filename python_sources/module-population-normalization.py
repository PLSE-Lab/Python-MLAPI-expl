# %% [code]
import pandas as pd
import numpy as np

# path_to_inputs = '../kaggle/input/covid19-granular-demographics-and-times-series/'
# input_filename0 = 'departments_static_data.csv'
# dfpop = pd.read_csv(path_to_inputs+input_filename0, delimiter=',')
# input_filenameNORM="population_used_for_normalization.csv"

def def_pop_cols(dfpop):
    ## get all columns which relate to population
    pop_cols = []
    for col in dfpop.columns[2:]:
        if col[:3] == "Pop":
            pop_cols.append(col)
    return pop_cols

def get_sex_age(col):
    ## reads the column names and extract the tags are integers
    if "sex=H" in col:
        sex=1
    elif "sex=F" in col:
        sex=2
    elif "sex=all" in col:
        sex=0
    else:
        print("weird: ", col)
        sex=0

    if "age" in col:
        if "agemin" in col:
            agemin = int(col.split("agemin=")[1].split("_")[0])
            if "agemax" in col:
                agemax = int(col.split("agemax=")[1].split("_")[0])
            else:
                print("VERY VERY weird !", col)
        elif "age=all" in col:
            agemin=0
            agemax=150
        else:
            print("weird: ", col)
            agemin=0
            agemax=150
    else:
        print("a little but not really weird: ", col)
        agemin=0
        agemax=150
    return sex, agemin, agemax

def get_sub_pop_corresponding_to_col(col, pop_cols, dfpop):
    ## returns the corresponding sub-population (correct denominator) of any column.
    ## If no exact match is found, returns the global population
    sex, agemin, agemax = get_sex_age(col)
    token = False
    for pop_col in pop_cols:
        Psex, Pagemin, Pagemax = get_sex_age(pop_col)
        if Psex == sex :
            if Pagemin == agemin:
                if Pagemax == agemax:
                    pop = dfpop[["code", pop_col]] ## we use the appropriate sub-pop as denominator
                    token = True
                    break

    if token == True:
        pass
    else:
        ## no match: we divide by the total pop of the dept (all sex, all age)
        print("no match: ", col)
        pop = dfpop[["code", "Pop_sex=all_age=all_Population"]]
    return pop

def normalize(dfIN, dfpop):
    pop_cols = def_pop_cols(dfpop)
    df3 = dfIN.copy() # pd.DataFrame()
    for col in df3.columns[2:]:
        ## Nbre columns are divided by the correct sub-pop.
        if col[:4] == "Nbre" :
            pop = get_sub_pop_corresponding_to_col(col, pop_cols, dfpop)
            for date in df3.date.unique():
                df3.loc[(df3.date==date), col] = (df3.loc[df3.date==date, col]).values / pop.iloc[:,1].values
            df3 = df3.rename(columns={col: "Rate"+col[4:]})
    return df3

def invert_norm(Rate_cols, dfpop):
    pop_cols = def_pop_cols(dfpop)
    pops =[]
    for col in Rate_cols:
        if col[:4] == "Rate" :
            pop = get_sub_pop_corresponding_to_col(col, pop_cols, dfpop)
            pops.append(pop.sort_values(by='code').iloc[:,1].values)
        #     for date in df3.date.unique():
        #         df3.loc[(df3.date==date), col] = (df3.loc[df3.date==date, col]).values / pop.iloc[:,1].values
        # df3 = df3.rename(columns={col: "Rate"+col[4:]})
    return np.array(pops).transpose()
