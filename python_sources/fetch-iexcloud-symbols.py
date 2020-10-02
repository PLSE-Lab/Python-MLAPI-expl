from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
IEX_CLOUD_KEY = user_secrets.get_secret("IEX_CLOUD_KEY")

import pandas as pd

import requests
import csv

url = 'https://cloud.iexapis.com/v1'
SYMBOLS = '/ref-data/symbols'

next_ = url + SYMBOLS

fields = (
"symbol",
"name",
"date",
"type",
"region",
"currency"
)

with open('symbols.csv', 'w') as file:
    writer = csv.writer(file)
    writer.writerow(fields)
    response = requests.get(url + SYMBOLS,
                            params={'token': IEX_CLOUD_KEY})
    json = response.json()
    for sym in json:
        writer.writerow([ sym[field] for field in fields ])
