#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install ibm_watson wget')


# In[ ]:


with open("../input/private-cred/cred.txt","r") as cred:
    url1 = cred.readlines()[0]
with open("../input/private-cred/cred.txt","r") as cred:
    api1 = cred.readlines()[1]
with open("../input/private-cred/cred.txt","r") as cred:
    url2 = cred.readlines()[2]
with open("../input/private-cred/cred.txt","r") as cred:
    api2 = cred.readlines()[3]
    


# In[ ]:


url1 = url1.rstrip()
url2 = url2.rstrip()
api1 = api1.rstrip()
api2 = api2.rstrip()


# In[ ]:


from ibm_watson import SpeechToTextV1
import json
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator


# In[ ]:


authenticator = IAMAuthenticator(api1)
s2t = SpeechToTextV1(authenticator=authenticator)
s2t.set_service_url(url1)


# In[ ]:


get_ipython().system('wget -O shape_of_you.mp3  http://s-a4446748.mp3pro.xyz/6022b59c59e7ebb7d8265/Shape%20of%20You%20%28Acoustic%29.mp3')


# In[ ]:


filename='shape_of_you.mp3'


# In[ ]:


with open(filename, mode="rb")  as wav:
    response = s2t.recognize(audio=wav, content_type='audio/mp3')


# In[ ]:


recognized_text=response.result['results']


# In[ ]:


recognized_text


# In[ ]:


recognized_text=response.result['results'][0]["alternatives"][0]["transcript"]


# In[ ]:


from ibm_watson import LanguageTranslatorV3


# In[ ]:


version_lt='2018-05-01'
url_lt = url2
apikey_lt=api2


# In[ ]:



authenticator = IAMAuthenticator(apikey_lt)
language_translator = LanguageTranslatorV3(version=version_lt,authenticator=authenticator)
language_translator.set_service_url(url_lt)
language_translator


# In[ ]:


from pandas.io.json import json_normalize
language_translator.list_identifiable_languages()
json_normalize(language_translator.list_identifiable_languages().get_result(), "languages")


# In[ ]:


language_translator.list_identifiable_languages().get_result()


# In[ ]:


translation_response = language_translator.translate(    text=recognized_text, model_id='en-de')
translation_response


# In[ ]:


translation=translation_response.get_result()
translation


# ## this is the original translation

# Ein Club ist am besten vor uns an der Bar platziert, wo ich hingehe

# ## exact match

# In[ ]:





# In[ ]:




