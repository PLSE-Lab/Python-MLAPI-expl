#!/usr/bin/env python
# coding: utf-8

# # Purpose
# I've noticed there's several kernels tagged `data cleaning` that are excellent so I thought the best way I could contribute was to document what has been covered by each kernel.  Since this is "data for good" I think it's better if any of us can get closer to the City of LA's goals for this data and sharing can get us there faster.
# 
# Also some quick observations:
# 
# * I thought it was intersting that some Kagglers start with a dataframe with all columns with null values that they then replace with actual values (see an [example](https://www.kaggle.com/jazivxt/a-freakonomics-opportunity)) this looks like a nice way to stay organized
# * There are lots of fields that weren't identified by the City of LA in their data dictionary, it'll be interesting to see we can make of the new additions, I've listed those below too.
# 
# 
# 

# # Fields listed in City of LA Data Dictionary
# This table follows the columns listed in:
# 
# `city-of-la\CityofLA\Additional data\kaggle_data_dictionary`
# 
# | Field Name             | [tyagit3][1]          | [sobrinomario][2] | [shahules][3]            | [shrivastava][4]                            | [jazivxt][5]       | [karthickaravindan][6] | [danielbecker][7]      | [heena34][8]    | [bachrr][9]   |
# |------------------------|-----------------------|----------------|--------------------------|---------------------------------------------|--------------------|------------------------|------------------------|-----------------|---------------|
# | FILE_NAME              | FILE_NAME             |                | File Name                | file_name                                   | FILE_NAME          |                        | file                   | FILE_NAME       | File Name     |
# | JOB_CLASS_TITLE        | JOB_CLASS_TITLE       | job_position   | Position                 | title                                       | JOB_CLASS_TITLE    | Title                  | job_title              | JOB_CLASS_TITLE | Job Title     |
# | JOB_CLASS_NO           | JOB_CLASS_NO          | class_code     |                          | class_code                                  | JOB_CLASS_NO       | Class_code             | class_code             | JOB_CLASS_NO    | Class Code    |
# | REQUIREMENT_SET_ID     | REQUIREMENT_SET_ID    |                |                          |                                             | REQUIREMENT_SET_ID |                        |                        |                 |               |
# | REQUIREMENT_SUBSET_ID  | REQUIREMENT_SUBSET_ID |                |                          |                                             |                    |                        |                        |                 |               |
# | JOB_DUTIES             | JOB_DUTIES            | duties         | duties                   | duties (process_note duplicated this field) | JOB_DUTIES         | Duties                 | duties_text            | JOB_DUTIES      | DUTIES        |
# | EDUCATION_YEARS        |                       |                |                          |                                             |                    |                        |                        |                 |               |
# | SCHOOL_TYPE            |                       |                |                          |                                             |                    |                        |                        |                 |               |
# | EDUCATION_MAJOR        | EDUCATION_MAJOR       |                |                          |                                             | EDUCATION_MAJOR    |                        |                        |                 |               |
# | EXPERIENCE_LENGTH      |                       |                |                          |                                             |                    |                        |                        |                 |               |
# | FULL_TIME_PART_TIME    |                       |                |                          |                                             |                    |                        |                        |                 |               |
# | EXP_JOB_CLASS_TITLE    |                       |                |                          |                                             |                    |                        |                        |                 |               |
# | EXP_JOB_CLASS_ALT_RESP |                       |                |                          |                                             |                    |                        |                        |                 |               |
# | EXP_JOB_CLASS_FUNCTION |                       |                | requirements             | requirements                                |                    | Requirment             | requirements_text      |                 | REQUIREMENTS  |
# | COURSE_COUNT           |                       |                |                          |                                             |                    |                        |                        |                 |               |
# | COURSE_LENGTH          |                       |                |                          |                                             |                    |                        |                        |                 |               |
# | COURSE_SUBJECT         |                       |                |                          |                                             |                    |                        |                        |                 |               |
# | MISC_COURSE_DETAILS    |                       |                |                          |                                             |                    |                        |                        |                 |               |
# | DRIVERS_LICENSE_REQ    |                       |                |                          |                                             |                    |                        |                        |                 |               |
# | DRIV_LIC_TYPE          |                       |                |                          |                                             |                    |                        |                        |                 |               |
# | ADDTL_LIC              |                       |                |                          |                                             |                    |                        |                        |                 |               |
# | EXAM_TYPE              |                       |                |                          |                                             | EXAM_TYPE          |                        |                        |                 |               |
# | ENTRY_SALARY_GEN       | ENTRY_SALARY_GEN      | salary         | salary_start, salary_end | salary                                      | ENTRY_SALARY_GEN   | Annual_salary          | salary_from, salary_to |                 | ANNUAL SALARY |
# | ENTRY_SALARY_DWP       | ENTRY_SALARY_DWP      |                |                          | salary (combined with above)                | ENTRY_SALARY_DWP   | (combined with above)  |                        |                 |               |
# | OPEN_DATE              | OPEN_DATE             |                | opendate                 | open_date                                   | OPEN_DATE          | Open_date              | open_date              |                 | Open Date     |
# 
# [1]:https://www.kaggle.com/tyagit3/starter-text-bulletins-to-dataframe
# [2]:https://www.kaggle.com/sobrinomario/city-of-la-starter-kernel
# [3]:https://www.kaggle.com/shahules/discovering-opportunities-at-la
# [4]:https://www.kaggle.com/shrivastava/data-cleaning-is-one-tough-job
# [5]:https://www.kaggle.com/jazivxt/a-freakonomics-opportunity
# [6]:https://www.kaggle.com/karthickaravindan/preparing-dataset
# [7]:https://www.kaggle.com/danielbecker/l-a-jobs-data-exctraction-eda
# [8]:https://www.kaggle.com/heena34/extracting-some-values-using-regex
# [9]:https://www.kaggle.com/bachrr/dsfg-jobs-at-la-eda

# # Additional fields found by Kagglers
# 
# | Type                 | [tyagit3][1] | [sobrinomario][2] | [shahules][3] | [shrivastava][4]     | [jazivxt][5] | [karthickaravindan][6] | [danielbecker][7]           | [heena34][8] | [bachrr][9]          |
# |----------------------|--------------|-------------------|---------------|----------------------|--------------|------------------------|-----------------------------|--------------|----------------------|
# | application method   |              |                   |               | apply                |              | Where_to_Apply         | where_to_apply_text         |              | WHERE TO APPLY       |
# | application method   |              |                   |               |                      |              |                        | where_to_apply_notes        |              |                      |
# | application deadline |              |                   | deadline      | application_deadline |              | Application_deadine    | application_deadline_text   |              | APPLICATION DEADLINE |
# | application deadline |              |                   |               |                      |              |                        | application_deadline_review |              |                      |
# | application deadline |              |                   |               |                      |              |                        | application_deadline_notes  |              |                      |
# | selection process    |              |                   | selection     |                      |              |                        | selection_process_text      |              | SELECTION PROCESS    |
# | selection process    |              |                   |               |                      |              |                        | selection_process_notes     |              |                      |
# | selection process    |              |                   |               |                      |              |                        | selection_process_notice    |              |                      |
# | examination notes    |              |                   |               | promotional_exam     |              |                        |                             |              |                      |
# | salary               |              |                   |               |                      |              |                        | salary_flatrated:           |              |                      |
# | salary               |              |                   |               |                      |              |                        | salary_additional           |              |                      |
# | salary               |              |                   |               |                      |              |                        | salary_notes                |              |                      |
# | job duties           |              |                   |               |                      |              |                        | duties_notes                |              |                      |
# | requirements         |              |                   |               |                      |              |                        | requirements_text           |              |                      |
# | requirements         |              |                   |               |                      |              |                        | requirements_notes          |              |                      |
# | requirements         |              |                   |               |                      |              |                        | requirements_certifications |              |                      |
# 
# 
# [1]:https://www.kaggle.com/tyagit3/starter-text-bulletins-to-dataframe
# [2]:https://www.kaggle.com/sobrinomario/city-of-la-starter-kernel
# [3]:https://www.kaggle.com/shahules/discovering-opportunities-at-la
# [4]:https://www.kaggle.com/shrivastava/data-cleaning-is-one-tough-job
# [5]:https://www.kaggle.com/jazivxt/a-freakonomics-opportunity
# [6]:https://www.kaggle.com/karthickaravindan/preparing-dataset
# [7]:https://www.kaggle.com/danielbecker/l-a-jobs-data-exctraction-eda
# [8]:https://www.kaggle.com/heena34/extracting-some-values-using-regex
# [9]:https://www.kaggle.com/bachrr/dsfg-jobs-at-la-eda

# In[ ]:




