# See: https://www.kaggle.com/ajrwhite/covid-19-clinical-trials-api-access

import requests
import json
import pandas as pd

BASE_URL = 'https://clinicaltrials.gov/api/query/study_fields?expr=COVID-19&max_rnk=1000&fmt=json'

extract_fields = [
    'BriefTitle',
    'DesignAllocation',
    'DesignMasking',
    'DesignMaskingDescription',
    'InterventionName',
    'InterventionType',
    'LastKnownStatus',
    'OfficialTitle',
    'OutcomeAnalysisStatisticalMethod',
    'OutcomeMeasureTimeFrame',
    'SecondaryOutcomeMeasure',
    'StartDate',
    'StudyFirstPostDate',
    'StudyFirstPostDateType',
    'StudyFirstSubmitDate',
    'StudyFirstSubmitQCDate',
    'StudyPopulation',
    'StudyType',
    'WhyStopped'
]

def run_query(base_url = BASE_URL,
              extract_fields = extract_fields,
              expr = 'COVID-19',
              max_rnk = 1000,
              fmt = 'json'):
    
    if max_rnk < 1 or max_rnk > 1000:
        raise ValueError('max_rank should be in range 1 - 1000')
        
    if fmt not in ['json', 'csv']:
        raise ValueError('fmt must be "json" or "csv"')
    
    query_url = f'{base_url}&expr={expr}&max_rnk={max_rnk}&fmt={fmt}&fields={",".join(extract_fields)}'

    r = requests.get(query_url)
    
    # Handle failed request by raising an error
    if r.status_code != 200:
        r.raise_for_status()
    
    return json.loads(r.content)


def de_list(input_field):
    if isinstance(input_field, list):
        if len(input_field) == 0:
            return None
        elif len(input_field) == 1:
            return input_field[0]
        else:
            return '; '.join(input_field)
    else:
        return input_field
    
    
def json_to_df(j, de_list_df=True):
    df = pd.DataFrame(j['StudyFieldsResponse']['StudyFields'])
    if de_list_df:
        for col in df.columns:
            df[col] = df[col].apply(de_list)
    df['StudyFirstPostDate'] = pd.to_datetime(df.StudyFirstPostDate)        
    
    return df