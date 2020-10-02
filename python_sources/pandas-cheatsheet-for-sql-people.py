#!/usr/bin/env python
# coding: utf-8

# # Pandas cheatsheet for SQL people
# 
# Here we go.
# 
# *The blogpost at hackernoon was based on this notebook: https://hackernoon.com/pandas-cheatsheet-for-sql-people-part-1-2976894acd0*

# # Import modules and load data

# In[ ]:


import pandas as pd
import numpy as np


# In[ ]:





# In[ ]:


users = pd.read_csv("../input/users.csv")
courses = pd.read_csv("../input/courses.csv")
attendace = pd.read_csv("../input/attendance.csv")


# # Lets see the sample data from each dataset

# In[ ]:


users.head()


# In[ ]:


courses.head()


# In[ ]:


attendace.head()


# # Basic SELECT

# ## ```SELECT * FROM users```

# In[ ]:


users


# ## ```SELECT * FROM users LIMIT 0,10```

# In[ ]:


users[0:10]


# ## ```SELECT * FROM users WHERE email IS NULL```

# In[ ]:


users[users["email"].isnull()]


# ## ```SELECT first_name, last_name FROM users```

# In[ ]:


users[["first_name", "last_name"]]


# ## ```SELECT DISTINCT birth_year FROM users```

# In[ ]:


users[["birth_year"]].drop_duplicates()


# # Basic math & arithmetics

# ## ```SELECT AVG(points) FROM users```

# In[ ]:


users["points"].mean()


# ## ```SELECT SUM(points) FROM users```

# In[ ]:


users["points"].sum()


# ## ```SELECT * FROM users WHERE birth_year BETWEEN 1998 AND 2018```

# In[ ]:


users[(users["birth_year"]>=1998) & (users["birth_year"]<=2018)]


# # Using LIKE

# ## ```SELECT * FROM users WHERE first_name LIKE 'Ch%'```

# In[ ]:


users[users["first_name"].str.startswith('Ch')]


# ## ```SELECT * FROM users WHERE first_name LIKE '%es'```

# In[ ]:


users[users["first_name"].str.endswith('es')]


# ## ```SELECT * FROM users WHERE first_name LIKE '%on%'```

# In[ ]:


users[users["first_name"].str.contains('es')]


# ## ```SELECT first_name, last_name FROM users WHERE first_name LIKE '%on%'```

# In[ ]:


users[users["first_name"].str.contains('es')][["first_name", "last_name"]]


# <h2>```SELECT * FROM attendance atn
# LEFT JOIN users usr ON atn.user_id = usr.id```</h2>

# In[ ]:


at_users = pd.merge(attendace[["user_id", "course_id"]], users, how='left', left_on='user_id', right_on='id')


# In[ ]:


at_users


# Now lets join the above with course titles as the result it will be sama as the result of following SQL command
# 
# <h2>```SELECT * FROM attendance atn
# LEFT JOIN users usr ON atn.user_id = usr.id
# LEFT JOIN courses co ON co.id = atn.course_id```</h2>

# In[ ]:


course_user = pd.merge(at_users, courses, left_on="course_id", right_on="id")


# In[ ]:


course_user


# Now we can choose only necessary columns

# In[ ]:


course_user[["first_name", "last_name", "birth_year", "points", "course_name", "instructor"]]


# ## Order by

# ## ```SELECT * FROM users ORDER BY first_name, last_name```

# In[ ]:


users.sort_values(["first_name", "last_name"])


# ## ```SELECT * FROM users ORDER BY first_name, last_name DESC```

# In[ ]:


users.sort_values(["first_name", "last_name"], ascending=False)


# <h2>```SELECT first_name, last_name, birth_year,
# points, course_name, instructor FROM attendance atn
# LEFT JOIN users usr ON atn.user_id = usr.id
# LEFT JOIN courses co ON co.id = atn.course_id
# ORDER BY first_name, last_name```</h2>

# In[ ]:


pd.merge(
    pd.merge(
        attendace[["user_id", "course_id"]], users, how='left', left_on='user_id', right_on='id'
    ), courses, left_on="course_id", right_on="id")[
    ["first_name", "last_name", "birth_year", "points", "course_name", "instructor"]
].sort_values(["first_name", "last_name"])


# In[ ]:




