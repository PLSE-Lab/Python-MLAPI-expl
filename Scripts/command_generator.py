import pandas as pd

kernels = pd.read_csv("Kernels.csv")
kernelsversions = pd.read_csv("KernelVersions.csv")
"""
kernelsversions.KernelLanguageId == 2 
Specifies the Language ID
2 is pure Python
8 is Jupyter Notebook
"""
selectedkernels = kernels[kernels.Id.isin(kernelsversions[kernelsversions.KernelLanguageId == 2].KernelId)].sort_values(
    'TotalVotes', ascending=False).head(100000)
users = pd.read_csv("Users.csv")
"""
Creates batch file, this one is for macOS
"""
with open("get_kernels.command", "w+") as file:
    for i, j in selectedkernels.iterrows():
        commandstrings = "kaggle kernels pull " + users[
            users.Id == j.AuthorUserId].UserName + "/" + j.CurrentUrlSlug + "\n"
        file.writelines(commandstrings)
        commandstrings = "sleep 2\n"
        file.writelines(commandstrings)
    file.close()
