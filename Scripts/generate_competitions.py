import pandas as pd
from urllib.request import urlopen
from urllib.error import URLError
from bs4 import BeautifulSoup

kernels = pd.read_csv("Kernels.csv")
kernelsversions = pd.read_csv("KernelVersions.csv")
selectedkernels = kernels[kernels.Id.isin(
    kernelsversions[kernelsversions.
        ScriptLanguageId.isin({2, 8})].ScriptId)]. \
    sort_values('TotalVotes', ascending=False) \
    .head(100000)
users = pd.read_csv("Users.csv")
competition_count = 0
with open("kernels_competitions.csv", "w+", encoding="utf-8") as file:
    for i, j in selectedkernels.iterrows():
        competition_count += 1
        uid = users[users.Id == j.AuthorUserId]
        if uid.size > 0:
            url = 'https://www.kaggle.com/' + uid['UserName'].iloc[0] + '/' + j.CurrentUrlSlug
            try:
                webpage = urlopen(url)
                soup = BeautifulSoup(webpage, "lxml")
                description = (soup.find("meta", property="og:description")).encode('utf-8', errors='ignore').strip()
                competition = description.split("Using data from ".encode('utf-8').strip(), 1)[1][
                              :-29] if description else "No competition"
                competition = competition.decode('utf-8')
            except (URLError, UnicodeEncodeError) as e:
                competition = "No competition"
            file.writelines(j.CurrentUrlSlug + "," + competition + "\n")
            print(competition_count)
    file.close()
