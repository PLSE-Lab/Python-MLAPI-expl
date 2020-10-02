#!/usr/bin/env python
# coding: utf-8

# ## In which language do people have to fix stuff more often?
# 
# In every commit a developer changes one or more files. Some of the commits add new functionality, and some are fixing bugs added before. So I was wondering files of which language tend to have more fixes, compared to total number of changes. I detect the language by file extension, and if it's a fix or not by certain words in the commit message. This is of course not too reliable, but hopefull still would give a good estimation.

# In[23]:


import pandas as pd
import bq_helper
github = bq_helper.BigQueryHelper(active_project='bigquery-public-data', dataset_name='github_repos')


# In[24]:


query = """
WITH file_changes AS (
    SELECT
        diff.new_path AS file_path,
        commit.commit AS commit_sha,
        -- mark a changed file as 'is_fix' when the commit message looks like a fix of
        -- some problem: contains a full word 'fix', 'fixing', 'fixes' or 'bug', case insensitive
        CASE
            WHEN REGEXP_CONTAINS(commit.message, r"(?i)(^|[^\w])bug([^\w]|$)") THEN 1
            WHEN REGEXP_CONTAINS(commit.message, r"(?i)(^|[^\w])fix(ed|ing|es)?([^\w]|$)") THEN 1
            ELSE 0
        END AS is_fix
    FROM `bigquery-public-data.github_repos.commits` commit
    JOIN UNNEST(commit.difference) diff
    -- exclude merge commits, as they aggregate changes from multiple commits,
    -- and often contain all those commits' messages
    WHERE ARRAY_LENGTH(commit.parent) <= 1)
SELECT
    -- extract the file path part after the last dot, with limited
    -- length, so that it's more probable to be a real file extension
    LOWER(REGEXP_EXTRACT(file_path, r"\.(\w{1,15})$")) AS file_extension,
    COUNT(is_fix) AS n_total_changes,
    SUM(is_fix) AS n_fix_changes,
    COUNT(DISTINCT(commit_sha)) AS n_commits
FROM file_changes
GROUP BY file_extension
HAVING
    -- include only popular file extensions: with many files and many changes in GitHub
    n_total_changes > 10000
    AND n_commits > 10000
-- take the most buggy file extensions
ORDER BY n_fix_changes / n_total_changes DESC
LIMIT 2000
"""
print('Estimated query size: %.1f GB' % github.estimate_query_size(query))


# The function `BigQueryHelper.query_to_pandas_safe` fails if it takes more than 30 seconds for the query to finish, but in my case it sometimes takes up to 20 minutes to execute on the full dataset (`commits` table). The timeout is hardcoded there, so I created a helper function to increase the timeout.

# In[25]:


def query_with_custom_timeout(bq_helper_instance, query, timeout_seconds):
    query_job = bq_helper_instance.client.query(query)
    rows = list(query_job.result(timeout=timeout_seconds))
    if len(rows) > 0:
        return pd.DataFrame(data=[list(x.values()) for x in rows],
                            columns=list(rows[0].keys()))
    else:
        return pd.DataFrame()


# In[26]:


get_ipython().run_cell_magic('time', '', 'file_extension_stats = query_with_custom_timeout(github, query, timeout_seconds=3600)')


# In[27]:


def set_fixes_ratio(df):
    df['fixes_ratio'] = 100 * df.n_fix_changes / df.n_total_changes
set_fixes_ratio(file_extension_stats)
file_extension_stats.head(20)


# As you can see, there are a lot of peculiar file extensions in the result set like `.dmm`, `.vapi`,  `.skipped`, which are not easy to map to a programming language.  So let's filter out unknown extensions and only leave those that may be related. I composed a list of all mentioned programming languages from the [Stack Overflow Developer Survey Results 2017](https://insights.stackoverflow.com/survey/2017#technology), added a couple more obvious ones like HTML and CSS, and then took a list of file extensions for each language from the Wikipedia. As a result I've got the mapping below. Note that I had to merge _C_ and _Objective-C_ into one because their file extensions almost fully overlap.

# In[40]:


popular_languages = {
    'Assembly': '.asm',
    'C or Objective-C': '.c .h .mm',
    'C#': '.cs',
    'C++': '.cc .cpp .cxx .c++ .hh .hpp .hxx .h++',
    'Clojure': '.clj .cljs .cljc .edn',
    'CoffeeScript': '.coffee .litcoffee',
    'Common Lisp': '.lisp .lsp .l .cl .fasl',
    'Dart': '.dart',
    'Elixir': '.ex .exs',
    'Erlang': '.erl .hrl',
    'F#': '.fs .fsi .fsx .fsscript',
    'Go': '.go',
    'Groovy': '.groovy',
    'Haskell': '.hs .lhs',
    'Java': '.java',
    'JavaScript': '.js',
    'Julia': '.jl',
    'Lua': '.lua',
    'Matlab': '.m',
    'Perl': '.pl .pm .t .pod',
    'PHP': '.php .phtml .php3 .php4 .php5 .php7 .phps .php-s',
    'Python': '.py .pyc .pyd .pyo .pyw .pyz ',
    'R': '.r .RData .rds .rda',
    'Ruby': '.rb',
    'Rust': '.rs .rlib',
    'Scala': '.scala .sc',
    'Smalltalk': '.st',
    'SQL': '.sql',
    'Swift': '.swift',
    'TypeScript': '.ts .tsx',
    'VB.NET': '.vb',
    'VBA': '.vba',
    'Shell': '.sh',
    'HTML': '.html',
    'CSS': '.css',
    'Text': '.txt .md .markdown',
    'Kotlin': '.kt'
}


# In[41]:


popular_languages_df = pd.DataFrame(
    [(lang, extension.replace('.', '').strip().lower())
     for (lang, extensions) in popular_languages.items()
     for extension in extensions.split()],
    columns=['language', 'file_extension']
)
popular_languages_df.describe()


# Note that with this table each file extension corresponds to one language. Now let's merge the language names with the data queried from the dataset, grouping together the extensions that belong to the same languge.

# In[42]:


languages_stats = file_extension_stats    .merge(popular_languages_df, left_on='file_extension', right_on='file_extension')    .groupby('language')    .sum()
# the fixes_ratio column has to be recalculated now based on summed up n_total_changes and n_fix_changes
set_fixes_ratio(languages_stats)
languages_stats = languages_stats.reset_index().sort_values('fixes_ratio', ascending=False)
languages_stats.index = range(len(languages_stats.index))
languages_stats


# So this is it, the list of languages sorted by ratio of "fix" commit against the total amount of commits! Almost all the languages got represented in the query, except for VBA. Below the same information is displayed on a bar plot.

# In[45]:


import seaborn as sns
import matplotlib.pyplot as plt
f, ax = plt.subplots(figsize=(12, 10))
sns.barplot(data=languages_stats,
            x="fixes_ratio", y="language",
            palette="Reds_r")
ax.set(ylabel="",
       xlabel="Percentage of commits that fix something")
sns.despine(left=True, bottom=True)


# #### SQL gets fixed most often
# As you see on the plot SQL files get fixed considerably more often than any other language: more than 18%. So almost every 5th commit to SQL files is a fix of some problem rather than development of a new feature. C++ follows as the second with a noticeable difference from the rest with fixes ratio more than 15%.
# 
# #### Better off with Java rather than Scala?
# At my work I hear discussions often if we should write new code in Java or Scala. Now there's some data to help decide: Java code gets fixed less than 10% of the times, while Scala is more than 12%. But of course Java is also a lot better represented language, there are 12 times more commits in Java code rather than in Scala.
# 
# #### Front-end code is not so bad
# Surprisingly for me the front-end files of standard extensions get fixed the least: JavaScript, CSS and HTML . One could think that changes like "Fix CSS of some button for yet another browser" would be made very often, but according to this data this is not true. Seems like "Fix SQL query for yet another database" is something that happens a lot more often.

# ### Which languagues are the most active ones?
# We already have the number of commits per language in the table, so let's make that visual too.

# In[46]:


f, ax = plt.subplots(figsize=(12, 10))
sns.barplot(data=languages_stats.sort_values('n_commits', ascending=False),
            x="n_commits", y="language",
            palette="Blues_r")
ax.set(ylabel="",
       xlabel="Number of commits")
sns.despine(left=True, bottom=True)


# The text and Markdown files ended up being changed the most. Maybe this is because almost all GitHub repositories have a `README.md` file with some documentation in it, and surely it gets changed a lot.
# 
# "C or Objective-C" follows on the second place. Maybe a big contribution to that number is the [Linux repository](https://github.com/torvalds/linux) with a huge history of commits.
