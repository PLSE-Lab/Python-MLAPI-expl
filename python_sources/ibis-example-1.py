import ibis

print("Let's see what tables we have to work with:\n")
con = ibis.sqlite.connect("../input/database.sqlite")
for table in con.list_tables():
    print(" - " + table)

users = con.table("Users")
print("\n\nWhat columns does the Users table have?\n")
print(users.info())

print("\n\nLet's see what users have made the most submissions:\n")
submissions = con.table("Submissions")
print(submissions.group_by("SubmittedUserId")
                 .aggregate(submissions.Id.count().name("NumSubmissions"))
                 .sort_by(("NumSubmissions", False))
                 .limit(20)
                 .join(users, [("SubmittedUserId", "Id")])
                 .execute()[["DisplayName", "NumSubmissions"]])
