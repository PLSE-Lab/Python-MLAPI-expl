import pandas as pd


kernels = pd.read_csv("Kernels.csv")
kernelsversions = pd.read_csv("KernelVersions.csv")
selectedkernels = kernels[kernels.Id.isin(
    kernelsversions[kernelsversions.
                    ScriptLanguageId.isin({2, 8})].ScriptId)].\
                    sort_values('TotalVotes', ascending=False)\
    .head(100000)


users = pd.read_csv("Users.csv")
with open("get_kernels.cmd", "w+", encoding="utf-8") as file:
    for i,j in selectedkernels.iterrows():
        commandstrings="kaggle kernels pull " +            users[users.Id==j.AuthorUserId].UserName +             "/" +j.CurrentUrlSlug +"\n"
        file.writelines(commandstrings)
        commandstrings="TIMEOUT 2\n"
        file.writelines(commandstrings)
    file.close()
    

