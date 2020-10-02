#!/usr/bin/env python
# coding: utf-8

# <h1>Working in LA</h1>
# <p>
#     The city of LA has partned with Kaggle in order to induldge clever insights on dealing with it's predicted hiring challenge.<br/>
#     In this kernel I will analyse the problem and the data given in order that we can fathom the situation and come to solutions.    
#     <br/>
#     The desired outcome will be to draw practical solutions which will draw in a more diverse and of higher quality applicant pool for the LA job postings. And, to find heuristics to make job promotions/hierarchies more acessible.
# </p>
# <p>
#     <strong>My objectives for the kernel will be to:</strong>
#     <ul>
#         <li>Evidence the problem and prove it's validity</li>
#         <li>If the problem is valid:</li>
#         <ul style="margin:3px">
#             <li>Draw insights from the data provided</li>
#             <li>Examine options that may induldge a more diverse range of candates for job postings</li>
#         </ul>
#     </ul>
# </p>
# <p style="margin-top:5px">
#     <strong>The steps taken in this kernel are the proceding:</strong>
# </p>
#     <ol>
#         <li>Extraction and structuring the data.</li>
#         <li>Validation of the problem</li>
#         <li>Analysis of the given data</li>
#         <li>Analysis of the given problem</li>
#         <li>Creation and validation of hypothesis.</li>
#    </ol>
# 

# <hr/>

# <h2>Extracting and Structuring The Data</h2>

# Packages:

# In[ ]:


# Packages
import os, re, datetime
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# directories
bulletin_dir = "../input/cityofla/CityofLA/Job Bulletins/"
titles = pd.read_csv('../input/cityofla/CityofLA/Additional data/job_titles.csv')


# Extraction of salaries:

# In[ ]:


def find_salary(lines):
    """
        Searches a salary pattern in the array lines
        input: array of strings
        output: str
    """
    for line in lines:
        flat_rate = re.search('(?i)flat|rated', line)
        if flat_rate and flat_rate[0]:
            salary = re.search('\$\d+,\d+', line)
            if salary:
                return salary[0] + ' (flat-rated)'

        else:
            ranges = re.split('and|;|,\s', line)
            for range in ranges:
                bounds = re.findall('\$\d+,\d+', range)
                if len(bounds) == 1: # in case of a flat-rated salary which was not properly noted
                    return bounds[0] + ' (flat-rated)'
                if len(bounds) == 2: # found a range.
                    return '-'.join(bounds) 


# Extraction of exam type:

# In[ ]:


def find_exam_type(line):
    """
        Find's the exam type of the job bulletin
        input: string
        output: string
    """
    exam_type = []
    found = re.findall(r'OPEN COMPETITIVE BASIS|INTERDEPARTMENTAL PROMOTIONAL|DEPARTMENTAL PROMOTIONAL BASIS', line, re.IGNORECASE) 

    if "OPEN COMPETITIVE BASIS" in found:
        exam_type.append("OPEN")

    if "INTERDEPARTMENTAL PROMOTIONAL" in found:
        exam_type.append("INT_DEPT_PROM")

    if "DEPARTMENTAL PROMOTIONAL BASIS" in found:
        exam_type.append("DEPT_PROM")

    exam_type = ', '.join(exam_type)
    return exam_type


# Gathers all the needed data of a job bulletin:

# In[ ]:


def gather_data(df):
    """
    Gathers all the needed data of a job bulletin contained in a datafram
    input: pandas.DataFrame
    output: list of objects
    """
    
    data_reqs = []
    data = {}
    default = {
        'JOB_CLASS_TITLE': '',
        'ENTRY_SALARY_GEN': None,
        'ENTRY_SALARY_DWP': None,
        'JOB_DUTIES': '',
        'EXAM_TYPE': ''
    }
    
    test = df[0].str.contains('(?i)requirements')

    for idx, line in enumerate(df[0]):     
        if idx == 0:
            # Title
            title_regex = re.search('\w([^\s]{2,}[^\n|\t])+\S', line)
            if title_regex:
                data['JOB_CLASS_TITLE'] = title_regex[0].strip()
            
        # Class Code
        code_line_reg = re.search('(?i)Class\s+Code:.*', line)        
        if code_line_reg:     
            code = re.search('\d+', code_line_reg[0])
            if code:
                data['JOB_CLASS_NO'] = code[0]
                
        
        # Salary
        s1_salary_line =  re.search('ANNUAL\s?SALARY', line)
        if s1_salary_line:
            section = get_section(df, idx + 1)
            s1 = find_salary(section)
            if s1:
                data['ENTRY_SALARY_GEN'] = s1
        
        s2_salary_line = re.search('(?i).*Department.*Water.*Power.*', line)
        if s2_salary_line:
            s2 = find_salary([s2_salary_line[0]])
            if s2:
                data['ENTRY_SALARY_DWP'] = s2
                 
        # Open Date
        open_date_line = re.search('(?i)((Open Date)|(Date)):.*', line)
        if open_date_line:
            open_date = re.search('(?<=:)[^\(\)]*', open_date_line[0])
            data['OPEN_DATE'] = open_date[0].strip()
            
        # Duties
        if "DUTIES" in line:
            section = get_section(df, idx + 1)
            data['JOB_DUTIES'] = ''.join(section)
            
        if 'APPLICATION DEADLINE' in line:
            deadline_section = get_section(df, idx + 1)
            data['DEADLINE_DATE'] = find_deadline(deadline_section) 
            
        # exam type
        if 'THIS EXAMINATION IS TO BE GIVEN' in line.upper():
            exam = find_exam_type(df[0][idx + 1])
            if exam:
                data['EXAM_TYPE'] = exam

        # Requirements and Education / Courses / Experience / Extra requirements ... 
        req_regex = re.search('REQUIREMENT', line)    
        if req_regex:
            reqs, notes = get_section(df, idx + 1, notes=True)
            req_sets = extract_sets(reqs)
            requirements = extract_extra_reqs(req_sets, notes)


    for idx, req in enumerate(requirements):  #iterate over the requirement sets
        entry = { 'REQUIREMENT_SET_ID': idx + 1, 'REQUIREMENT_SUBSET_ID': 'A' }
        entry.update(default)
        entry.update(data)
        entry.update(req)
        data_reqs.append(entry)          

    return data_reqs


# Extraction of deadline date:

# In[ ]:


def find_deadline(lines):
    """
    Find the job application deadline
    input: array of strings
    output: string
    """

    for line in lines:
        date = re.search("(?i)(MONDAY|TUESDAY|WEDNESDAY|THURSDAY|FRIDAY|SATURDAY|SUNDAY).*\d{4}", line)
        if date:
            return date[0]


# Extracts driver license:

# In[ ]:


def find_driver_license(text):
    """
        Searches for a driver license requirement and the type of the license required
        input: string
        output: tuple of strings
    """
    
    may_req = re.search("may[^\.]*requir[^\.]*driver[^\.]*license", text)
    req = re.search('(requir[^\.]*driver[^\.]*license)|(driver[^\.]*license[^\.]*requir)', text)
    
    if may_req:
        driver  = 'M'
    elif req:
        driver = 'R'
    else:
        driver = ''
        
    driver_types = []    
    if driver == 'M' or driver == 'R':
        driver_types = re.findall('(?i)class\s\w', text)
        driver_types = list(dict.fromkeys(driver_types)) # Removes any duplicates
        
    driver_types = ', '.join(driver_types)
    
    return driver, driver_types


# Extracts other need licenses:

# In[ ]:


def find_license(text):   
    """
    Finds additional licenses
    input: str
    output: str
    """
    text = text.lower()
    if "license" in text:
        if "driver's" not in text:
            look_behind = re.search('([A-Z]+\w*\s)+license', text)
            look_ahead = re.search('license \w+ ([A-Z]+\w*\s)+', text)
            if look_behind:
                return look_behind[0].upper()
            if look_ahead:
                return look_ahead[0].upper()
    if 'medical certificate' in text:
        return 'Medical Certificate'


# Extract the different sets of the requirement section:

# In[ ]:


def extract_sets(req):
    """Extract requirements sets and subsets (D1, D2)"""
    mandatories = []
    optionals = []
    
    previous_conditional = False
    for line in req:
        or_conditional = re.search('(?i)or\s?$', line)
        
        if previous_conditional:
            idx = len(optionals) - 1
            optionals[idx].append(line)
        elif or_conditional:
            optionals.append([line])
        else:
            mandatories.append(line)
            
        previous_conditional = True if or_conditional else False
    
    # This loop we is so we create combinatories of the optional requirements
    # e.g: we might have optionals: [[a, b],[c, d]]. We would then generate ac, ad, bc, bd.
    req_sets = []
    for idx, optional in enumerate(optionals):
        for conditional in optional:
            requirements = mandatories.copy()
            requirements.append(conditional)
            
            for idx2, optional_2 in enumerate(optionals):
                if idx == idx2:
                    continue
                for conditional2 in optional_2:
                    requirements.append(conditional2)
            req_sets.append(requirements)
    if not len(req_sets):
        req_sets = [mandatories]
    return req_sets  


# Extracts and gathers all the data in the 'requirement' section:

# In[ ]:


def extract_extra_reqs(req_sets, notes):
    """
        Extracts requirements such as:
        Education, Driver's Liscence, Expirence, Course Info, etc.
        (F, G, H, I, J, L, M, N, O, P1, P2, Q)
        input: 
            req_sets: array of strings
            notes: array of strings
        output: list of objects            
    """
    requirements = []
    driver = ''
    course_count = 0
    licenses = []
    
    # Extract driver's license and other licenses from notes
    for line in notes:
        if not driver:
                driver, driver_types = find_driver_license(line)
                license = find_license(line)
                if license:
                    licenses.append(license)

    
    for req_set in req_sets:
        data = {}
        
        # REQUIREMENT_TEXT
        data['REQUIREMENT_TEXT'] = ' | '.join(req_set)
        
        for line in req_set:
            
            # EDUCATION_YEARS, SCHOOL TYPE, EDUCATION_MAJOR
            education = re.search('(?i).*(education)|(college)|(school)|(university).*', line)
            if education:
                edu_type = re.search('(?i)(university)|(college)|(school)', line)
                if 'college or university' in line.lower():
                    data['SCHOOL_TYPE'] = 'COLLEGE OR UNIVERSITY'
                elif edu_type:
                    data['SCHOOL_TYPE'] = edu_type[0].upper()
                    
                major = re.search('((degree)|(major)) in(\s[A-Z]+\w*)+', line)
                if major:
                    data['EDUCATION_MAJOR'] = re.search('(\s[A-Z]+\w*)+', major[0])[0].upper()
                
                years = re.search('(?i)((\d+)|(\w)+).(years?)', line)
                if years:
                    data['EDUCATION_YEARS'] = years[0].upper()
                    
                semesters = re.search('(?i)((\d+)|(\w)+).(semesters?)', line)
                if semesters:
                    numb = re.search('\d+', semesters[0])
                    if numb:
                        data['EDUCATION_YEARS'] = numb[0]

    
            # EXPERIENCE_LENGTH
            experience = re.search('(?i).*experience.*', line)
            if experience:

                length = re.search('(?i)((\d+)|(\w+)) years?', experience[0])
                if length:
                    data['EXPERIENCE_LENGTH'] = length[0].upper()

                
                time = re.search('(?i)((full)|(part))-?\s?time', experience[0])
                if time:
                    data['FULL_TIME_PART_TIME'] = time[0].upper()
                    
                title = re.search('((as an?)|(at the level of an?))(\s[A-Z]+\w*)+', experience[0])
                if title:
                    title = re.search('(\s[A-Z]+\w*)+', title[0])[0]
                    data['EXP_JOB_CLASS_TITLE'] = title.upper()

            # DRIVER'S LICENSE            
            if not driver:
                driver, driver_types = find_driver_license(line)
                
            # OTHER LICENSES
            license = find_license(line)
            if license:
                licenses.append(license)
                
            # courses
             #COURSE_LENGTH COURSE_SUBJECT MISC_COURSE_DETAILS, COURSE_COUNT
            if re.search("(?i)courses?", line):
                title = re.search('([A-Z]+\w*\s)+course', line)
                if title:
                    course_count += 1
                    data['COURSE_SUBJECT'] = title[0]
                misc = re.search("course.*is equivalent to", line)
                if misc:
                    misc = re.search("(?=equivalent to).*", line)
                    data['MISC_COURSE_DETAILS'] = misc[0]
                
                course_length = re.search(
                    '((\d+)|(\w+))\s((semesters?)|(quarters?)|(years?)(hours?))(?:.{0,50}course)', line)
                if course_length:
                    data['COURSE_LENGTH'] = course_length[0]

        licenses = list(dict.fromkeys(licenses))
        data['COURSE_COUNT'] = course_count if course_count else None
        data['ADDTL_LIC'] = ', '.join(licenses)
        data['DRIVERS_LICENSE_REQ'] = driver
        data['DRIV_LIC_TYPE'] = driver_types
        requirements.append(data)
    
    return requirements


# Extracts all text bellow a header:

# In[ ]:


def get_section(df, idx, notes=False):
    """Extracts the section bellow a header"""
    ln = df[0][idx]
    sentences = []
    
    while not ln.isupper(): 
        if ln:
            sentences.append(ln)
        idx += 1
        ln = df[0][idx]
    
    if notes and re.search('(?i)note', ln):
        notes = get_section(df, idx + 1)
        return sentences, notes
    elif notes:
        return sentences, []
    
    return sentences


# Creates the CSV:

# In[ ]:


def get_csv(directory):
    values = []

    for idx, filename in enumerate(os.listdir(directory)):
        with open(bulletin_dir + "/" + filename, 'r', errors='ignore') as f:
            lines = f.readlines()

            df = pd.DataFrame(lines)
            df = df.replace('\n','', regex=True).replace('\t', '', regex=True)
            text = ' '.join(lines)
            
            data_reqs = gather_data(df)
            for entry in data_reqs:
                entry['FILE_NAME'] = filename      
                values.append(entry)
                
    data_df = pd.DataFrame(values)
    data_df['OPEN_DATE'] = pd.to_datetime(data_df['OPEN_DATE'])
    data_df = data_df[['FILE_NAME', 'JOB_CLASS_TITLE', 'JOB_CLASS_NO', 
                       'REQUIREMENT_SET_ID', 'REQUIREMENT_SUBSET_ID', 
                       'JOB_DUTIES', 'EDUCATION_YEARS', 'SCHOOL_TYPE', 
                       'EDUCATION_MAJOR', 'EXPERIENCE_LENGTH', 'FULL_TIME_PART_TIME',  
                       'EXP_JOB_CLASS_TITLE', 'COURSE_LENGTH', 'COURSE_SUBJECT', 
                       'MISC_COURSE_DETAILS', 'COURSE_COUNT',
                       'DRIVERS_LICENSE_REQ', 'DRIV_LIC_TYPE', 'ADDTL_LIC', 
                       'ENTRY_SALARY_GEN', 'ENTRY_SALARY_DWP', 'OPEN_DATE', 
                       'DEADLINE_DATE','EXAM_TYPE', 'REQUIREMENT_TEXT']]
    return data_df


# <h1>Final CSV File</h1>

# In[ ]:


output = get_csv(bulletin_dir)


# In[ ]:


output.head()


# <h2>Dictionary and the output</h2>
# <p>Now after doing all this dirty work let's update our dictionary to fit our csv.</p>
# <p>We will add our new values 'deadline_date' and 'requirements_text'</p>

# In[ ]:


dictionary =  pd.read_csv('../input/cityofla/CityofLA/Additional data/kaggle_data_dictionary.csv')

columns = dictionary.columns
new_data = pd.DataFrame([['DEADLINE_DATE', 'U', 'Deadline of for the job bulletin', 'String', '', 'Yes', ''],
                      ['REQUIREMENT_TEXT', 'V', 'The requirements of a particular job', 'String', '', 'Yes', '']], 
                     columns=columns)

dictionary = dictionary.append(new_data)
dictionary.tail(2)


# In[ ]:


dictionary.to_csv('dictionary.csv', index=False)
output.to_csv('bulletins.csv', index=False)


# <hr/>

# <h1>Analysis</h1>

# <h3>Proving The Problem</h3>
# <p>
# The first step in our crusade on getting californians a job is to evaluate the problem.
# Is the problem real? Does LA face a hiring challenge?
# </p>
# <p>
# First, I want to show case <a href="https://www.agc.org/sites/default/files/Files/Communications/2018_Workforce_Survey_California.pdf">a survey</a> by Autodesk made with construction firms. Which provides some interesting results for us.
# </p>
# <img width="700" src="https://i.imgur.com/r9mKuIC.png" style="margin:30px">
# <p>
# The graphs gives an optimistic outlook on the construction market given that most construction firms surveyed expect to expand their companies but, coming back to the topic of this kernel, they struggle to find the workforce to do it.<br/>
# As a matter of fact most companies do not expect that the situation will get better:
# <img src="https://i.imgur.com/Ca2tD86.png" width="600" style="margin:40px"/>
# Other sources seem to corroborate with this claim, here is a <a href="https://www.pasadenastarnews.com/2018/01/03/california-construction-firms-plan-to-hire-in-2018-but-skilled-workers-in-short-supply/">news article</a>, from Pasadena Star-News, that showcases another survey with similar results.
# <p>
# But these are only a single example in the construction field in the state of California, which is not representative of the whole workforce ecosystem of Los Angeles.<br/>
# Let us move to more generalized evidence.
# </p>
# <p>
# Of particular interests for us, is <a href="https://www.ppic.org/content/pubs/report/R_1208DRR.pdf">the paper</a> of Deborah Reed, which asserts that the supply of college-educated workers will mismatch it's demands and the scenario will be aggravated by the retirement of baby boomers.
# <img src="https://i.imgur.com/HqMlA7d.png" width="500px">
# </p>
# <p>
# With this analysis we conclude our proof.<br/>
# Unsurprisingly, we reach the conclusion that the government of LA is right.<br/>
# Labour shortage is a problem.
# </p>

# In[ ]:


# section packages
import matplotlib.pyplot as plt
import seaborn as sns
data = output
unique = data.drop_duplicates(['JOB_CLASS_NO'])


# <h2>Analysis of the Data</h2>
# <p>Now, we will proceed, to make considerations on the data provided.

# Starting out, we were provided 682 job bulletins, we formated them into 1213 rows (some jobs get multiple rows each with a different range of requirements).</br>
# We will make analysis on the formated version with 1213 rows and with the formated version with any duplicated row excluded.

# <h3>
# Let's examine the number of Job Bulletins per year. 
# </h3>

# In[ ]:


dates = pd.DataFrame([[date.year, 1] for date in pd.to_datetime(unique['OPEN_DATE'])])
dates = dates.groupby([0]).aggregate(sum)
sns.set(style="darkgrid")
# Plot the responses for different events and regions
plt.plot(dates[1].keys(), dates[1].values)


# <p>
#    There is a monotnic increase of the number of job bulletins.
# </p>
# <p>
#    <small>The pitfall is due to the low ammount of job bulletins in 2019, which were collected in just a small fraction of 2019.</small>
# </p>
# <p>
#     We may assume that the increase of job bulletins is directly related to an increase of open positions in the market.
#     </p>

# <h3>
#     Which are the types of positions that are most demanded.
# </h3>

# In[ ]:


pos = [val for val in unique['JOB_CLASS_TITLE'] if type(val) == type('')]
d = pd.DataFrame([[val.split()[-1], 1] for val in pos if val])
d = d.groupby([0]).aggregate(sum)
counts = d.nlargest(15, 1)[1]
labels = counts.keys()

# plt.figure(figsize=(10, 8))
# plt.title('Type of Job x Bulletin Count')
plot = sns.barplot(x=counts, y=labels)


# <p>
#     We notice the need of very sophisticated workers, such as engineers, technicians, specialists, etc. Positions that require a college degree, which corroborates with out previous research on the shortage of educated workers.
# </p>
# <p>
#     Maybe of bigger magnitude in our data is the need of managerial positions (exemplified by the inflated number of supervisor, manager and superintendent bullentins). We may assume that, from our data, most of the demanded positions require highly educated workers or workers already higher up in hierarchy.
# </p>
# <p>
#  Conversely we also admit some bias. As most jobs will have a managerial stage it's expected that their bulletins would be more common.
# </p>

# <h3>Salaries</h3>

# In[ ]:


def clean_salary(text):
    sal = re.split(' |-', text)
    start = round(int(sal[0][1:].replace(',', '')), -3) // 1000
    end = round(int(sal[1][1:].replace(',','')), -3) // 1000 if sal[1][0] == '$' else start
    avg = (start + end) // 2
    diff = end - start
    return start, end, avg, diff

def get_salaries(data):
    start_sal = []
    end_sal = []
    avgs = []
    for sal in data:
        if not sal:
            continue        
        start, end, avg, _ = clean_salary(sal)
        start_sal.append(start)
        end_sal.append(end)
        avgs.append(avg)
    return pd.DataFrame(data={'start': start_sal, 'end': end_sal, 'avg': avgs })

gen = get_salaries(unique['ENTRY_SALARY_GEN'])
dwp = get_salaries(unique['ENTRY_SALARY_DWP'])
gen= gen.sort_values(by='start')
dwp = dwp.sort_values(by='start')


# In[ ]:


f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex='col', sharey='row', figsize=(14,8))

sns.distplot(gen['start'], kde=False, ax=ax1)
sns.distplot(gen['end'], kde=False, ax=ax1)
ax1.set_title('Salary Ranges')
ax1.set(xlabel='Salary (in thousands)', ylabel='Frequency')
ax1.legend(labels=['Lower Bound', 'Upper Bound'])

sns.distplot(dwp['start'],kde=False, ax=ax2)
sns.distplot(dwp['end'], kde=False, ax=ax2)
ax2.set(xlabel='Salary (in thousands)', ylabel='Frequency')
ax2.set_title('Salary Distribution DWP')
ax2.legend(labels=['Lower Bound', 'Upper Bound'])

sns.distplot(gen['avg'], kde=False, ax=ax3, color='g')
ax3.set(xlabel='Salary (in thousands)', ylabel='Frequency')
ax3.set_title('Averaged Salary Distribution')

sns.distplot(dwp['avg'], kde=False, ax=ax4, color='g')
ax4.set(xlabel='Salary (in thousands)', ylabel='Frequency')
ax4.set_title('Averaged Salary Distribution DWP')

print('Average across salaries:', round(np.mean(gen['avg'])))
print('Average across salaries (DWP):', round(np.mean(dwp['avg'])))


# Plotted above is the distribution of the salary ranges of all jobs. <br/>
# Bellow is the distribution of the salaries with the ranges averaged.<br/>
# <p>
# We see that the deviation of DWP salaries is much bigger from the as it's graphs are much more evenly distributed.<br/>
# Interestingly DWP and city jobs both average on the same salary, even though in a per job basis DWP has a higher salary.
# </p>   
# 

# <h3>Highest payed jobs</h3>

# In[ ]:


res = []
for row in unique.iterrows():
    title = row[1][1]
    sal = row[1][19]
    if sal and title:
        start, end, avg, diff = clean_salary(sal)
        res.append([title, start, end, avg, diff])
res = pd.DataFrame(res)
largests = res.nlargest(10, 3)

plt.title('Top averaged salaries')
plt.xlabel('Salaries in thousands')
ax = sns.barplot(y=0, x=3, data=largests)
ax.set(xlabel='Averaged Salary (in thousands)', ylabel='Titles')
plt.show()


# <h3>Jobs with the highest deviations</h3>

# Highest differentials between the upperbound and lowerbound of the salary of a job:

# In[ ]:


plt.title('Highest deviations')
plt.xlabel('Salaries in thousands')
ax = sns.barplot(y=0, x=4, data=res.nlargest(10, 4))
ax.set(xlabel='Differential in thousands', ylabel='Titles')
plt.show()


# <h1>Analysis of the Problem</h1>
# <p>
# As was previously proved LA faces a hiring challenge, but luckily it does have capability for improvement.<br/>
#     <p>
#         Alternatively from the LA low official (U3) unemployment rates, it's U6 rates soar a bit.<br/>
#         Data from the federal <a href="https://www.bls.gov/lau/stalt.htm">Bureau Labor Statistics</a> reveals that LA has relatively high underemployment rate given it's high U6 rates, which is into the double digits.<br/>
#         (The U6 underployment rates differs from the U3 as they account for "marginally attached" workers and parts of the population that gave up looking for jobs.)
# </p>
# 
# 
# <p>
#     In order to get this significant part of the population into the workforce I propose improving the outlet used for hiring, namely the job bulletins.<br/>
#     In the rest of this kernel I will explore ideas that might improve the job bulletins and assuage the task of applicant reading them.
#     

# <h2>Job Bulletins</h2>
# Here is an example of a job bulletin:

# In[ ]:


from wand.image import Image as Img
Img(filename='../input/cityofla/CityofLA/Additional data/PDFs/2018/September/Sept 28/ARTS MANAGER 2455 092818.pdf', resolution=300)


# <p>
#     I plowed some of these while extracting data for the csv files, and going through them felt as hassle.<br/>
#     These posting are long and dense.
# </p>
# <p>
#     As I only had to go through few of them it wasn't much of a problem, but for a candidate that may go through dozens of job postings a day, the long format is an inconvenience and may as well be an inhibitor.
# </p>
# <p>
#     Making job postings that relies on the candidate motivation is not the most suitable approach.<br/>
#     We should attempt to make the task of reading a job posting satisfying.<br/>
#     If we can't to that, we want to make it at least the least unsatisfying as possible.
# </p>

# <h3>Better formats</h3>
# <p>
#     We can improve the format of the posting by simply making it a little bit cleaner, adding more padding, and separating sections better.
#     <br/>
# Here is the same job posting reformatted:
# </p>
# <img src="https://i.imgur.com/IKIipgH.png">

# <p>It feels lighter, the text is more readable. It provides a slightly improved QOL for the candidate.</p>

# <h2>Text Readability</h2>
# <div style="width:50%;display: inline-block; margin:3%">
# The <a href="https://app.readable.com/text/?demo">readable.com</a> website provides a text analysis utility, which measures the readbility of text.
# <p>
# Sampling the same job bulletin as before, we get the rating D.
# </p>
# <p>
#     Besides the D score which it's meaning is self explanatory, we also receive some other ratings with a more abstract meaning.
# </p>
# <p>
#     In particular, the text is also tested for two Flesch-Kincaid redability tests, and the Gunning Fox Index test. The three tests have a single motif, that is to indicate how difficult a passage in English is to understand.
# </p>
# <p>
#     Our ratings (as shown bellow) infer that the bulletin text in most approprietly fit for highly educated workers.<br/>
#     Which of course is not the case for every applicant.<br/>
#     Further on we will explore if such predicament holds across all of the bulletins.
# </p>
# 
# </div>
# <img style="float:right; display: inline-block; width:40%" src="https://i.imgur.com/0lKYM3n.png">
# 
# <div style="display:inline-block; width:40%; margin:5%; text-align:center">
#     <div stlye="margin-top:25px">
#         Metrics for the Flesch-Kincaid Grad Level:
#         <img src="https://i.imgur.com/8WXSNVt.png"/>
#     </div>
#     <div style="margin-top:25px">
#         Metrics for the Flesh-Kincaid reading ease:
#         <img src="https://i.imgur.com/741a81f.png"/>
#     </div>
# </div> 
# <div style="display:inline-block; width:40%;margin:5%; text-align:center; float:right">
#         Metrics for the Gunning Fod Index:
#         <img src="https://i.imgur.com/vQal1Iz.png"/>
# </div>
# 

# <h3>Results on all data</h3>
# <p>We will check the whole dataset to see if this scores hold.</p>
# <p>I will be using a straight-forward python package for text complexity analysis, which provides a lot of different measures.<br/> we will only focus on the Flesch Reading Ease and Gunning Fox index.
# </p>
# More info on the package can be found <a href="https://pypi.org/project/readability/">here.</a>

# In[ ]:


# Notice: kaggle package library doesn't have the readability package
# The rest of the post proceeds with a screenshot of the output of the code
import readability
import os
def get_metrics(directory):
    results = []
    for idx, filename in enumerate(os.listdir(directory)):
        with open(directory + "/" + filename, 'r', errors='ignore') as f:
            lines = f.readlines()
            res = readability.getmeasures(lines, lang='en')
            res = res['readability grades']
            results.append(res)
    return pd.DataFrame(results)
results = get_metrics("./Job Bulletins/")

flesch = results['FleschReadingEase']
gunning = results['GunningFogIndex']

import matplotlib.pyplot as plt
import seaborn as sns
flesch_m = np.mean(flesch)
gunning_m = np.mean(gunning)

fig, (ax1, ax2) =plt.subplots(1,2, figsize=(20, 7))
ax1.title.set_text('Average Score')
sns.barplot(x=['Flesch Reading Ease', 'Gunning Fog Index'], y=[flesch_m, gunning_m], ax=ax1)

ax2.title.set_text('Distribution')


sns.distplot(gunning, color='orange', kde=False, rug=False, ax=ax2)
sns.distplot(flesch, kde=False, rug=False, ax=ax2)
plt.legend(labels=['Gunning Fog Index', 'Flesch Reading Ease'])
plt.show()

print('Average Flesch:', flesch_m.round(2),'\nAverage Gunning:', gunning_m.round(2))
print('Min/Max Flesch:', np.min(flesch).round(2), '/',np.max(flesch).round(2))
print('Min/Max Gunning:', np.min(gunning).round(2), '/', np.max(gunning).round(2))


# <img src="https://i.imgur.com/W1vhMNX.png">

# Our results across all data diverge a little from our previous sample. 
# <p>
# We have an mean Flesch Reading Ease score of: 38<br/>
# A Gunning Fog Index average of 19.45, which is quite amazing, it simply breaks our previous metrics.
# </p>
# <p>
#    The results are quite surprising, I was not expecting much variance between bulletins, due to the patternized format and and writing style.<br/>
#     But, instead we see quite a lot. Flesch test ranging from 14 to 68, and Gunnig test from 13 to an unbelievable 26.
# </p>
# <p>
#     The reason for this variance is that such metrics take into consideration and are highly influenced by the lenght of sentences, which does vary a lot.<br/>
#     Many bulletins have extremely long sentences of duties and requirements, which range far beyond paragraphs in regular writings. Which explain the off the charts scores.
# </p>
# 
# <small>Formula for Gunning index:</small>
# <img src="https://wikimedia.org/api/rest_v1/media/math/render/svg/84cd504cf61d43230ef59fbd0ecf201796e5e577">
# 
# <p>
#     <br/>
#     Despite the divergence, our results still conclude the same as before.<br/>
#     The text is fairly complex and not made for the general public.
# </p>

# <h4>Conclusion</h4>
# <p>
#     What our research shows is that the job postings are highly biased towards fluent and educated speakers. Which is not the case for every american.<br/>
#     In fact, California has one of highest state distribution of hispanics, and the LA County in particular contains more than 4.9 million hispanics.
#     <img height="200" width="300" src="https://www.pewhispanic.org/wp-content/uploads/sites/5/2013/08/PH-2013-08-latino-populations-1-01.png">
#     <br/>
#     The job postings should be more approachable for the general public.<br/>
#     Taking simple steps, as using a simpler vocabulary and writing shorter sentences will improve the readibility of the postings.
# </p>
# 
# 

# <h3>Solution</h3>
# 
# <div style="width:50%;display: inline-block; margin:3%">
#     <p>
#     After breaking big sentences into paragraph and replacing a few of the more complex words of the bulletin for more common ones, this is the result we get:
#     </p>
#     <p>
#      We have some marginal improvement.<br/>
#      Our Gunning Index is now two scales bellow.<br/>
#      And Flesch Grade Level is at the 'Average' scale<br/>
#      More important than the rating, the bulletin is actually easier to read.<br/>
#      Short paragraphs with simple writing is more pleasant to read.<br/>
#      This is an improvement both to the applicants which are not highly eductaed in english, 
#      as well as applicants that have to go through many of these in a daily basis.<br/>
#       This makes their task a bit easier, which in turn might mean they won't skip our job posting.
#     </p>
# 
# </div>
# <img style="float:right; display: inline-block; width:40%" src="https://i.imgur.com/a0CObuu.png">

# <h2>Text Length</h2>
# As mentioned previously, the overall text length of the bulletin may be an issue.<br/>
# 

# In[ ]:


import re
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
import numpy as np
def get_word_length(directory):
    results = []
    for idx, filename in enumerate(os.listdir(directory)):
        with open(directory + "/" + filename, 'r', errors='ignore') as f:
            text = ''.join(f.readlines())
            words = re.findall('\w+', text)
            results.append((len(words), len(text)))

    return pd.DataFrame(results)

lengths = get_word_length("../input/cityofla/CityofLA/Job Bulletins/")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,4))

sns.distplot(lengths[0], color='blue', ax=ax1, kde=False)
ax1.title.set_text('No. Words in job postings')
sns.distplot(lengths[1], color='orange', kde=False, rug=False, ax=ax2)
ax2.title.set_text('No. Characters in job postings')
plt.show()
print('Avg. number of words of the job postings: ', int(np.mean(lengths[0])))
print('Avg. number of characters: ', int(np.mean(lengths[1])))


# <p>
#     As shown, our job postings average on 9750 characters. Which without any point of reference is hard to say if this is or not ideal.
# </p>
# <p>
#     This <a href="https://www.ere.net/long-job-descriptions-and-titles-can-hurt-you-and-so-can-short-ones/">article</a> by Chris Foreman brings significant information on the issue.<br/>
#     It showcases the results of a ressearch about the ideal length of job descriptions.<br/>
#     The data used in the research was aquired by tracking the click-to-apply ratio of online job applications.<br/>   
#  </p>   
#  <p>
#      The study, proposes that ideally job descriptions should not be longer than 10,000 characters nor shorter than 2,000.<br/>
#     <img style="display:block; width: 50%;" src="https://s3.amazonaws.com/media.eremedia.com/uploads/2015/02/job-description-click-to-apply.png">
#     <p>Using the standard above, it can be affirmed, that the length of the job bulletins are suboptimal, and that better results may be reached if we reduce their length.</p>
#     <p>The article also contains insights which I quite agree:</p>
#     <blockquote>
# Too-long job descriptions may suggest a stifling working environment and demand too much effort from candidates who already are investing a great deal of time in their job search. 
# </blockquote>
#  </p>

# <p>
#     Backed by the analysis above, my proposal is to keep only the minimal necessary information on the job bulletins (pratically the aim is to average at 7,000 chareacters). We delegate to the hiring team the task of informing about the job in depth. This approach is better bacause we aleviate the load of offshore candidates, which in turn may lead to more fruitful outcomes. 
# </p>
# <p>
#     I simplified our sample bulletin, deleting some of the more subaltern information contained in the duties and notes sections.
#     </p>

# <h2>Final Conclusions</h2>
# <p>
#     As intended, this kernel focused on applicable solutions that would improve the size and diversity of the pool of candidates in the L.A. job openings. This led me to try to improve the quality of life of the applicant by making the job bulletins simpler.
# </p>
# <p>
#     We made a few straight-forward modifications to the job postings.<br/>
#     Namely, the format of the postings was changed to a more attractive design with more spacing so it's easier to read.<br/>
#     We broke the text down to smaller paragraphs and changed complex words for common ones so the text is easy to understand and it's less biased torwards fluent english speakers.<br/>
#     We reduced the length of bulletin, so it's easier to get through it.
# </p>
# 
# <div style="width:45%;display:inline-block;">
#     Proposed Bulletin:<br />
#     <p>
#     https://i.imgur.com/xEIVrxO.jpg <br/>
#     </p>
#     <ul>
#         <li>Length: 8937 characters</li>
#         <li>Gunning Index: 12.5</li>
#         <li>Flesch-Kincade Grade Level: 11.4</li>
#         <li>Flesch Reading Ease: 36.2</li>
#    </ul>
# </div>
# 
# <div style="width:45%;display:inline-block;">
#     Original Bulletin:<br/>
#     <p>
#     https://imgur.com/KlCc36y<br/>
#     https://imgur.com/aUPoaxz<br/>
#     </p>
#    <ul>
#     <li>Length: 12203 characters</li>
#     <li>Gunning Index: 15.4</li>
#      <li>Flesch-Kincade Grade Level: 13.4</li>
#      <li>Flesch Reading Ease: 25.8</li>
#    </ul>
# </div>
# 
