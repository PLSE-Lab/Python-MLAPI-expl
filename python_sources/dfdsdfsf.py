
import pandas as pd
import sqlite3
con = sqlite3.connect('../input/database.sqlite')
a=pd.read_sql_query("""
SELECT s.TeamId, s.PublicScore,
c.CompetitionName AS Competition,
julianday(c.DeadLine) - julianday(s.DateSubmitted) as TimeToFinish 
FROM Submissions AS s
INNER JOIN Teams as t on (s.TeamId = t.Id)
INNER JOIN Competitions as c on (t.CompetitionId = c.Id)
INNER JOIN EvaluationAlgorithms AS e ON (c.EvaluationAlgorithmId = e.Id)
WHERE e.Abbreviation = 'CategorizationAccuracy'
OR e.Abbreviation = 'AUC'
ORDER BY Competition DESC, s.TeamId DESC, TimeToFinish DESC
""", con)
print(a)