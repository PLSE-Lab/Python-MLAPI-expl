"""
Many thanks to Rachael Tatman for the ideas and approaches presented 
in her "Getting Started with Automated Data Pipelines" series.

https://www.kaggle.com/rtatman/kerneld4769833fe
"""

import csv
import pandas as pd
import os
os.system('pip install csvvalidator')
from csvvalidator import *


def datetime_or_empty_string(format_):
    """
    Return a value check function which raises a ValueError if the supplied
    not empty value cannot be converted to a datetime using the supplied format string.
    See also `datetime.strptime`.
    """

    def checker(v):
        if v != '':
            datetime.strptime(v, format_)

    return checker


# CustomCSVValidator class contains method print_validation_results that prints validation results,
# stored in ValidationProblems list

class CustomCSVValidator(CSVValidator):

    def __init__(self, field_names):
        super().__init__(field_names)
        self.validation_problems = None

    def print_validation_summary(self, errors_to_display=10):
        problems = self.validation_problems
        problems_count = len(problems)

        if problems_count < errors_to_display:
            errors_to_display = problems_count

        print(f'\nFound {problems_count} validation errors')
        if problems_count > 0:
            print(f'\nFirst {errors_to_display} errors:')
            for i in range(errors_to_display):
                print(f"Exception number {problems[i]['code']}:{problems[i]['message']} row:{problems[i]['row']}",
                      end=" ")
                if 'column' in problems[i]:
                    print(f"column:({problems[i]['column']}) ", end="")
                    if 'field' in problems[i]:
                        print(f"\'{problems[i]['field']}\' ", end=" ")
                        if 'value' in problems[i]:
                            print(f"value:\'{str(problems[i]['value'])}\'", end="")
                print("")


class ProjectsValidator(CustomCSVValidator):

    def __init__(self, path_to_file, encoding='utf-8-sig'):
        field_names = ('rcn',  # int
                       'id',  # int
                       'acronym',
                       'status',  # SIGNED,CLOSED,TERMINATED
                       'programme',
                       'topics',
                       'frameworkProgramme',
                       'title',
                       'startDate',  # date
                       'endDate',  # date
                       'projectUrl',
                       'objective',
                       'totalCost',  # float
                       'ecMaxContribution',  # float
                       'call',
                       'fundingScheme',
                       'coordinator',
                       'coordinatorCountry',
                       'participants',
                       'participantCountries',
                       'subjects')
        super().__init__(field_names)
        self.path_to_file = path_to_file
        self.encoding = encoding

    def validate(self):
        # check file header
        self.add_header_check(code='H1', message='Header validation failed')
        # checks fields
        self.add_value_check('rcn',  # the name of the field
                             int,  # a function
                             'EX1',  # code for exception
                             '`rcn` must be an integer')  # message to report if error thrown

        self.add_value_check('id',
                             int,
                             'EX2',
                             '`id` must be an integer')

        self.add_value_check('endDate',
                             datetime_or_empty_string('%Y-%m-%d'),  # date in specified format or empty string
                             'EX3',
                             'invalid `endDate`')

        self.add_value_check('startDate',
                             datetime_or_empty_string('%Y-%m-%d'),
                             'EX4',
                             'invalid `startDate`')

        self.add_value_check('status',  # only 'SIGNED','CLOSED','TERMINATED' are allowed
                             enumeration('SIGNED', 'CLOSED', 'TERMINATED'),
                             'EX5',
                             '`status` not recognized')

        self.add_value_check('totalCost',
                             match_pattern('^[0-9,]*$'),
                             'EX6',
                             '`totalCost` must be a float or empty')

        self.add_value_check('ecMaxContribution',
                             match_pattern('^[0-9,]+$'),
                             'EX7',
                             '`ecMaxContribution` must be a float')

        with open(self.path_to_file, encoding=self.encoding) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=';')
            self.validation_problems = super().validate(csv_reader)


class OrganizationsValidator(CustomCSVValidator):

    def __init__(self, path_to_file, encoding='utf-8-sig'):
        field_names = ('projectRcn',  # int not null
                       'projectID',  # int not null
                       'projectAcronym',
                       'role',  # only has 'participant', 'coordinator', 'partner', 'hostInstitution','beneficiary'
                       'id',  # int not null
                       'name',
                       'shortName',
                       'activityType',
                       'endOfParticipation',  # boolean: 'False' or 'True' not null
                       'ecContribution',  # float, can be null
                       'country',
                       'street',
                       'city',
                       'postCode',
                       'organizationUrl',
                       'vatNumber',
                       'contactForm',
                       'contactType',
                       'contactTitle',
                       'contactFirstNames',
                       'contactLastNames',
                       'contactFunction',
                       'contactTelephoneNumber',
                       'contactFaxNumber')
        super().__init__(field_names)
        self.path_to_file = path_to_file
        self.encoding = encoding

    def validate(self):
        # check file header
        self.add_header_check(code='H1', message='Header validation failed')
        # checks fields
        self.add_value_check('projectRcn',
                             int,
                             'EX1',
                             '`projectRcn` must be an integer')

        self.add_value_check('projectID',
                             int,
                             'EX2',
                             '`projectID` must be an integer')

        self.add_value_check('role',
                             enumeration('participant', 'coordinator', 'partner', 'hostInstitution', 'beneficiary'),
                             'EX3',
                             '`role` not recognized')

        self.add_value_check('id',
                             int,
                             'EX4',
                             '`id` must be an integer')

        self.add_value_check('endOfParticipation',
                             enumeration('false', 'true'),
                             'EX5',
                             '`endOfParticipation` must be boolean')

        self.add_value_check('ecContribution',
                             match_pattern('^[0-9,]*$'),
                             'EX6',
                             '`ecContribution` must be a float or empty')

        with open(self.path_to_file, encoding=self.encoding) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=';')
            self.validation_problems = super().validate(csv_reader)


class CountriesValidator(CustomCSVValidator):

    def __init__(self, path_to_file, encoding='utf-8-sig'):
        field_names = ('euCode',
                       'isoCode',
                       'name',
                       'language')
        super().__init__(field_names)
        self.path_to_file = path_to_file
        self.encoding = encoding

    def validate(self):
        # check file header
        self.add_header_check(code='H1', message='Header validation failed')

        with open(self.path_to_file, encoding=self.encoding) as csv_file:  # utf-8-sig removes BOM ufeff
            csv_reader = csv.reader(csv_file, delimiter=';')
            self.validation_problems = super().validate(csv_reader)


class ProgrammesValidator(CustomCSVValidator):

    def __init__(self, path_to_file, encoding='utf-8-sig'):
        field_names = ('rcn',
                       'code',
                       'title',
                       'shortTitle',
                       'language')
        super().__init__(field_names)
        self.path_to_file = path_to_file
        self.encoding = encoding

    def validate(self):
        # check file header
        self.add_header_check(code='H1', message='Header validation failed')
        # checks fields
        self.add_value_check('rcn',
                             int,
                             'EX1',
                             '`rcn` must be an integer')

        with open(self.path_to_file, encoding=self.encoding) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=';')
            self.validation_problems = super().validate(csv_reader)


# This is a dummy class for H2020 data file
class H2020DataFile:

    def __init__(self, path_to_file, encoding='utf-8-sig'):
        self.path_to_file = path_to_file
        self.validator = None
        self.encoding = encoding

    def validate(self):
        if not (self.validator is None):
            self.validator.validate()
        else:
            print('Validator object is not initialized.')

    def print_validation_summary(self, errors_to_display=10):
        if not (self.validator.validation_problems is None):
            self.validator.print_validation_summary(errors_to_display=errors_to_display)
        else:
            print('Validator object is not initialized. Please call validate() method')


class H2020Projects(H2020DataFile):
    """
    Works with cordis-h2020projects.csv file of Horizon 2020 Data
    
    Create object:
        p = H2020Projects('../input/cordis-h2020projects.csv', encoding='utf-8-sig')

    Validate cordis-h2020projects.csv:
        p.validate()
        
        This method checks a header and a format of columns
        
    Review validation results:
        p.print_validation_summary(errors_to_display = 10)
    
        `errors_to_display` -  number of errors to display in a report
        
    Read cordis-h2020projects.csv to a DataFrame
        df = p.read()

    """

    def __init__(self, path_to_file, encoding='utf-8-sig'):
        super().__init__(path_to_file, encoding=encoding)
        self.validator = ProjectsValidator(path_to_file, encoding=encoding)

    def read(self):
        if not (self.path_to_file is None):
            df = pd.read_csv(self.path_to_file, index_col='rcn', decimal=',', sep=';',
                             parse_dates=['startDate', 'endDate'], encoding=self.encoding)
            return df
        else:
            return None


class H2020Organizations(H2020DataFile):
    """
    Works with cordis-h2020organizations.csv file of Horizon 2020 Data
    
    Create object:
        p = H2020_Organizations('../input/cordis-h2020organizations.csv', encoding='utf-8-sig')

    Validate cordis-h2020organizations.csv:
        p.validate()
        
        This method checks a header and a format of columns
        
    Review validation results:
        p.print_validation_summary(errors_to_display = 10)
    
        `errors_to_display` -  number of errors to display in a report
        
    Read cordis-h2020organizations.csv to a DataFrame
        df = p.read()

    """

    def __init__(self, path_to_file, encoding='utf-8-sig'):
        super().__init__(path_to_file, encoding=encoding)
        self.validator = OrganizationsValidator(path_to_file, encoding=encoding)

    def read(self):
        if not (self.path_to_file is None):
            df = pd.read_csv(self.path_to_file, index_col='id', decimal=',', sep=';',
                             encoding=self.encoding, low_memory=False)
            df.ecContribution.fillna(0, inplace=True)
            return df
        else:
            return None


class H2020Countries(H2020DataFile):
    """
    Works with cordisref-countries.csv file of Horizon 2020 Data

    Create object:
        p = H2020Countries('../input/cordisref-countries.csv', encoding='utf-8-sig')

    Validate cordisref-countries.csv:
        p.validate()

        This method checks a header and a format of columns

    Review validation results:
        p.print_validation_summary(errors_to_display = 10)

        `errors_to_display` -  number of errors to display in a report

    Read cordisref-countries.csv to a DataFrame
        df = p.read()

    """

    def __init__(self, path_to_file, encoding='utf-8-sig'):
        super().__init__(path_to_file, encoding=encoding)
        self.validator = CountriesValidator(path_to_file, encoding=encoding)

    def read(self):
        if not (self.path_to_file is None):
            df = pd.read_csv(self.path_to_file, sep=';', encoding=self.encoding)
            return df
        else:
            return None


class H2020Programmes(H2020DataFile):
    """
    Works with cordisref-H2020programmes.csv.csv file of Horizon 2020 Data

    Create object:
        p = H2020Countries('../input/cordisref-H2020programmes.csv', encoding='utf-8-sig')

    Validate cordisref-H2020programmes.csv:
        p.validate()

        This method checks a header and a format of columns

    Review validation results:
        p.print_validation_summary(errors_to_display = 10)

        `errors_to_display` -  number of errors to display in a report

    Read cordisref-H2020programmes.csv to a DataFrame
        df = p.read()

    """

    def __init__(self, path_to_file, encoding='utf-8-sig'):
        super().__init__(path_to_file, encoding=encoding)
        self.validator = ProgrammesValidator(path_to_file, encoding=encoding)

    def read(self):
        if not (self.path_to_file is None):
            df = pd.read_csv(self.path_to_file, sep=';', encoding=self.encoding, low_memory=False,
                             dtype={'rcn': 'Int64'}, index_col='rcn')
            return df
        else:
            return None
