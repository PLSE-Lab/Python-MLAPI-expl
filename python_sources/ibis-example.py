import ibis

# This makes it so expressions get executed immediately
ibis.options.interactive = True

con = ibis.sqlite.connect("../input/database.sqlite")
print(con.list_tables())

scorecard = con.table("scorecard")
print(scorecard[scorecard.INSTNM=="Duke Unversity"][["INSTNM","Year"]])

print("\n\n\nScorecard has many columns:\n")
print(scorecard.info())
