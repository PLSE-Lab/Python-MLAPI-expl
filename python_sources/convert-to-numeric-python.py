# ==========================================================================
# The original dataset has text fields where numeric or boolean are required
# ==========================================================================

# This script reads the csv file and does the conversion

import numpy as np
import pandas as pd

raw_cols = ['State',
 'Uninsured Rate (2010)',
 'Uninsured Rate (2015)',
 'Uninsured Rate Change (2010-2015)',
 'Health Insurance Coverage Change (2010-2015)',
 'Employer Health Insurance Coverage (2015)',
 'Marketplace Health Insurance Coverage (2016)',
 'Marketplace Tax Credits (2016)',
 'Average Monthly Tax Credit (2016)',
 'State Medicaid Expansion (2016)',
 'Medicaid Enrollment (2013)',
 'Medicaid Enrollment (2016)',
 'Medicaid Enrollment Change (2013-2016)',
 'Medicare Enrollment (2016)']
 
# citation:
# http://stackoverflow.com/questions/12432663/
# what-is-a-clean-way-to-convert-a-string-percent-to-a-float
def p2f(x):
    '''convert string percent to float'''
    return float(x.strip('%'))/100
def d2f(x):
    '''convert dollar to float'''
    return float(x.strip('$'))
def t2b(x):
    '''convert text to boolean'''
    return True if x == "True" else False

state_coverage = pd.read_csv(filepath_or_buffer="../input/states.csv",
                            converters={'Uninsured Rate (2010)':p2f,
                                        'Uninsured Rate (2015)':p2f,
                                        'Average Monthly Tax Credit (2016)':d2f,
                                        'State Medicaid Expansion (2016)':t2b})

state_coverage['Uninsured Rate Change (2010-2015)'] = state_coverage.iloc[:,2]-state_coverage.iloc[:,1]

print(state_coverage.dtypes)
print(state_coverage.head())