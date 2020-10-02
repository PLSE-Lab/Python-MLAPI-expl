#!/usr/bin/env python
# coding: utf-8

# # Download DFDC files from notebook (kernel, Colab, Jupyter...)
# 
# This kernels shows how you can download the multiple files of the dataset of this competition from any notebook.
# 
# Kudos for [lytic](https://www.kaggle.com/sorokin) for the idea of patching the kaggle client: [wget train files](https://www.kaggle.com/c/deepfake-detection-challenge/discussion/121194#695299)

# In[ ]:


get_ipython().system('pip install kaggle')


# In[ ]:


import re
import json
import os

import six


# ## Credentials
# To get the Kaggle API credentials, go to the 'Account' tab of your user profile (`https://www.kaggle.com/<username>/account`) and select 'Create API Token'. This will trigger the download of `kaggle.json`, a file containing your API credentials. Once you have these, you need to set them as environment variables, the easiest way is to directly read this file but you can also hardcode the credentials in the notebook (not very secure). If you want to use it in Colab for example, you can upload the file to your Goggle Drive and then mount and read it.

# In[ ]:


# Un-comment if you want to read the credentials from a file
# with open(os.path.expanduser("~/.kaggle/kaggle.json"), "rb") as f:
#     creds = json.load(f)
# os.environ["KAGGLE_USERNAME"] = creds["username"]
# os.environ["KAGGLE_KEY"] = creds["key"]

# For this notebook I will use fake values
os.environ["KAGGLE_USERNAME"] = "username"
os.environ["KAGGLE_KEY"] = "key"


# In[ ]:


from kaggle.api.kaggle_api_extended import KaggleApi


# In[ ]:


class CustomApiClient(KaggleApi):
    def competitions_data_download_file_with_http_info(self, id, file_name, **kwargs):  # noqa: E501
        """Download competition data file  # noqa: E501
        This method makes a synchronous HTTP request by default. To make an
        asynchronous HTTP request, please pass async_req=True
        >>> thread = api.competitions_data_download_file_with_http_info(id, file_name, async_req=True)
        >>> result = thread.get()
        :param async_req bool
        :param str id: Competition name (required)
        :param str file_name: Competition name (required)
        :return: Result
                 If the method is called asynchronously,
                 returns the request thread.
        """

        all_params = ['id', 'file_name']  # noqa: E501
        all_params.append('async_req')
        all_params.append('_return_http_data_only')
        all_params.append('_preload_content')
        all_params.append('_request_timeout')

        params = locals()
        for key, val in six.iteritems(params['kwargs']):
            if key not in all_params:
                raise TypeError(
                    "Got an unexpected keyword argument '%s'"
                    " to method competitions_data_download_file" % key
                )
            params[key] = val
        del params['kwargs']
        # verify the required parameter 'id' is set
        if ('id' not in params or
                params['id'] is None):
            raise ValueError("Missing the required parameter `id` when calling `competitions_data_download_file`")  # noqa: E501
        # verify the required parameter 'file_name' is set
        if ('file_name' not in params or
                params['file_name'] is None):
            raise ValueError("Missing the required parameter `file_name` when calling `competitions_data_download_file`")  # noqa: E501

        collection_formats = {}

        path_params = {}
        if 'id' in params:
            path_params['id'] = params['id']  # noqa: E501
        if 'file_name' in params:
            path_params['fileName'] = params['file_name']  # noqa: E501

        query_params = []

        header_params = {}

        form_params = []
        local_var_files = {}

        body_params = None
        # Authentication setting
        auth_settings = ['basicAuth']  # noqa: E501
        
        return self.api_client.call_api(
        '/c/{id}/datadownload/{fileName}', 'GET',
        path_params,
        query_params,
        header_params,
        body=body_params,
        post_params=form_params,
        files=local_var_files,
        response_type='Result',  # noqa: E501
        auth_settings=auth_settings,
        async_req=params.get('async_req'),
        _return_http_data_only=params.get('_return_http_data_only'),
        _preload_content=params.get('_preload_content', True),
        _request_timeout=params.get('_request_timeout'),
        collection_formats=collection_formats)


# In[ ]:


api = CustomApiClient()
api.authenticate()


# Un-comment any of the lines below to download the required, the filenames list contains all the available filenames.

# In[ ]:


filenames = [f"dfdc_train_part_{i:02}.zip" for i in range(50)]
# api.competition_download_file(competition="deepfake-detection-challenge", file_name="dfdc_train_part_00.zip")
# api.competition_download_file(competition="deepfake-detection-challenge", file_name=filenames[0])

