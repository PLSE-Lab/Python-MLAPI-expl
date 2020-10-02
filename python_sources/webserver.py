#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from flask import Flask

app = Flask(__name__)

@app.route('/')
def index():
    return "Hey <b>there</b>!<br>{ 6 * 7}"

app.run()


# In[ ]:




