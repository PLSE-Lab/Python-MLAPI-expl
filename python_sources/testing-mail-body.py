# The data comes both as CSV files and a SQLite database

import pandas as pd

# You can read a CSV file like this
mails = pd.read_csv("../input/Emails.csv")
subjects = mails['MetadataSubject']
body_texts = mails['ExtractedBodyText']

for subject, body_text in zip(subjects, body_texts):
    print(subject)
    print(body_text)
    print('')

# It's yours to take from here!