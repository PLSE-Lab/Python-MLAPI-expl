'''
Project
For this project you are given a file that contains some parking ticket 
violations for NYC.

(It's just a tiny extract!)

If you want the full data set, it's available here: 
https://www.kaggle.com/new-york-city/nyc-parking-tickets/version/2#

For this sample data set, the file is named:

nyc_parking_tickets_extract.csv
Your goals are as follows:

Goal 1
Create a lazy iterator that will return a named tuple of the data in each row. 
The data types should be appropriate - i.e. if the column is a date, you should 
be storing dates in the named tuple, if the field is an integer, then it should 
be stored as an integer, etc.

Goal 2
Calculate the number of violations by car make.

Note:
Try to use lazy evaluation as much as possible - it may not always be possible 
though! That's OK, as long as it's kept to a minimum.
'''


from collections import namedtuple, defaultdict
from functools import partial
from datetime import datetime


file_name = '../input/parking-tickets/nyc_parking_tickets.csv'


# creating headers and first row:
with open(file_name) as f:
    next(f)
    headers = next(f).strip('\n').split(',')
    sample_data = next(f).strip('\n').split(',')

# we want to use column_names as fields:
column_names = [header.replace(' ','_').lower() 
                  for header in headers]


'''
* 'summons_number'        -> int
* 'plate_id'              -> str
* 'registration_state'    -> str
* 'plate_type'            -> str
* 'issue_date'            -> date
* 'violation_code'        -> int
* 'vehicle_body_type'     -> str
* 'vehicle_make'          -> str
* 'violation_description' -> str
'''


Tickets = namedtuple('Tickets',column_names)

def read_data():
    with open(file_name) as f:
        next(f), next(f)
        yield from f

# functions for parsing data:
def parse_int(value, *, default=None):
    try:
        return int(value)
    except ValueError:
        return default
    
def parse_date(value, *, default=None):
    date_format = '%m/%d/%Y'
    try:
        return datetime.strptime(value,date_format).date()
    except ValueError:
        return default

def parse_string(value, *, default=None):
    try:
        cleaned = value.strip()
        if not cleaned:
            return default
        else:
            return cleaned
    except ValueError:
        return default

column_parsers = ( parse_int,                               # summons_number - default is None
                   parse_string,                            # plate_id - default is None
                   lambda x: parse_string(x,default=''),    # registration_state
                   partial(parse_string,default=''),        # plate_type
                   parse_date,                              # issue_date - default is None
                   parse_int,                               # violation_code - default is None
                   partial(parse_string,default=''),        # vehicle_body_type
                   parse_string,                            # vehicle_make - default is None
                   lambda x: parse_string(x,default='')     # violation_description
                  )


def parse_row(row, *, default=None):
    fields = row.strip('\n').split(',')
    # iterating two times
    parsed_data = [func(field)
                   for func,field in zip(column_parsers,fields)]
    # if there is any None-field:
    if all(item is not None for item in parsed_data):
        return Tickets(*parsed_data)
    else:
        return default