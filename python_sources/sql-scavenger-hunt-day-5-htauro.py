import bq_helper

github = bq_helper.BigQueryHelper(active_project="bigquery-public-data", dataset_name="github_repos")

query = """
            SELECT L.license, COUNT(sf.path) AS number_of_files
            FROM `bigquery-public-data.github_repos.sample_files` as sf
            INNER JOIN `bigquery-public-data.github_repos.licenses` as L
            ON sf.repo_name = L.repo_name
            GROUP BY L.license
            ORDER BY number_of_files DESC
            """
            
file_count_by_licenses = github.query_to_pandas_safe(query, max_gb_scanned=6)
file_count_by_licenses.head()

###########################

github.head("sample_files")
github.head("sample_commits")

query2 = """
        SELECT DISTINCT sc.repo_name, COUNT(sc.commit) AS num_commit
        FROM `bigquery-public-data.github_repos.sample_commits` AS sc
        INNER JOIN `bigquery-public-data.github_repos.sample_files` AS sf
        ON sf.repo_name = sc.repo_name
        WHERE sf.path LIKE '%.py'
        GROUP BY sc.repo_name
        ORDER BY num_commit DESC
        """
        
number_of_commits_per_repo_python = github.query_to_pandas_safe(query2, max_gb_scanned=6)

number_of_commits_per_repo_python.head()

############################

query3 = """
            WITH python_repos AS (
            SELECT DISTINCT repo_name
            FROM `bigquery-public-data.github_repos.sample_files`
            WHERE path LIKE '%.py'
            )
            
            SELECT sc.repo_name, COUNT(commit) as num_commits
            FROM `bigquery-public-data.github_repos.sample_commits` AS sc
            INNER JOIN python_repos
            ON sc.repo_name = python_repos.repo_name
            GROUP BY sc.repo_name
            ORDER BY num_commits DESC
            """
            
num_python_commits = github.query_to_pandas_safe(query3, max_gb_scanned = 6)

num_python_commits.head()