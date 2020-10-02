# Large part borrowed from: https://github.com/MuckRock/API-examples/blob/master/export_jurisdiction_stats.py

import os
import requests
import csv

from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
token = user_secrets.get_secret("MUCKROCK_TOKEN")
url = 'https://www.muckrock.com/api_v1/'

headers = {'Authorization': 'Token {}'.format(token), 'content-type': 'application/json'}
next_ = url + 'jurisdiction'

j_fields = (
"id",
"name",
"slug",
"abbrev",
"level",
"parent",
"absolute_url",
"average_response_time",
"fee_rate",
"success_rate"
)

page = 1
print("Getting Jurisdictions...")
with open('jurisdictions.csv', 'w') as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(j_fields)
    
    while next_ is not None:
        r = requests.get(next_, headers=headers)
        try:
            json = r.json()
            next_ = json['next']
            for datum in json['results']:
                csv_writer.writerow([datum[field] for field in j_fields])
            print("Page {} of {}".format(page, json['count'] / 20 + 1))
            page += 1
        except Exception as e:
            print(e)

a_fields = (
"id",
"name",
"slug",
"status",
"exempt",
"types",
"requires_proxy",
"jurisdiction",
"website",
"twitter",
"twitter_handles",
"parent",
"appeal_agency",
"url",
"foia_logs",
"foia_guide",
"public_notes",
"absolute_url",
"average_response_time",
"fee_rate",
"success_rate",
"has_portal",
"has_email",
"has_fax",
"has_address",
"number_requests",
"number_requests_completed",
"number_requests_rejected",
"number_requests_no_docs",
"number_requests_ack",
"number_requests_resp",
"number_requests_fix",
"number_requests_appeal",
"number_requests_pay",
"number_requests_partial",
"number_requests_lawsuit",
"number_requests_withdrawn"
)
page = 1
next_ = url + 'agency'
with open('agencies.csv', 'w') as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(a_fields)
    
    while next_ is not None:
        r = requests.get(next_, headers=headers)
        try:
            json = r.json()
            next_ = json['next']
            for datum in json['results']:
                csv_writer.writerow([datum[field] for field in a_fields])
            print("Page {} of {}".format(page, json['count'] / 20 + 1))
            page += 1
        except Exception as e:
            print(e)