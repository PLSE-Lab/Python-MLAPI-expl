#!/usr/bin/env python
# coding: utf-8

# This notebook has some notes on the `requests` API. It's primarily intended as a refresher.

# ### Async support
# * `requests` is synchronous. To perform HTTP queries in an asynchronous environment use `aiohttp` instead.
# 
# ### The most useful parameters
# 
# * To specify URL query strings use the `params` kwarg. E.g.:
# 
#   ```python
#   payload = {'key1': 'value1', 'key2': 'value2'}
#   r = requests.get('https://httpbin.org/get', params=payload)
#   # r.url == https://httpbin.org/get?key2=value2&key1=value1
#   ```
# 
# * `r.content` is the raw bytes, `r.text` is a decode attempt using a best-guess encoding (viewable/settable via `r.encoding`) for mimetypes that text-based. `r.json()` is a convenience function for converting to JSON.
# * To stream input set `stream=True` and use `iter_content(chunk_size=?)`:
#   
#   ```python
#   with open(filename, 'wb') as fd:
#     for chunk in r.iter_content(chunk_size=128):
#         fd.write(chunk)
#   ```
# 
#   Note that this will uncompress gzip and deflate transer encoded data beforehand, if you need the raw compressed bytes use `r.raw`.
# * HTTP headers are specified using the `headers` argument:
# 
#   ```python
#   requests.get(url, headers={'user-agent': 'my-app/0.0.1'})
#   ```
#   
#   For the authorization header use the `auth` parameter instead. See next section.
# 
# * `r.status_code` to get the status code. `r.raise_for_status()` to throw on a "bad" HTTP response status code.
# * `r.headers` has the headers, both getable and settable.
# * Disable redirects with `allows_redirects=False`. See the history of redirects with `r.history`.
# * Specify a timeout using `timeout`. This is recommended for production usage; `requests` can block indefinitely otherwise.
# 
# ### Authentication
# * Authentication flows are documented [here](https://requests.readthedocs.io/en/master/user/authentication/) in the docs. HTTP Basic Authentication, the most common form, is supported as follows:
# 
#     ```python
#     from requests.auth import HTTPBasicAuth
#     requests.get('https://api.github.com/user', auth=HTTPBasicAuth('user', 'pass'))
#     ```
#  
#   Note that in the HTTP header this looks like so: `Authorization: Basic YWxhZGRpbjpvcGVuc2VzYW1l`.
#   
# * More complicated authentication schemes are also supported, but for e.g. OAuth2 you'll need some side-library support, and more complex parameterization.
# 
# ### POST
# * POST data (via `request.post`) sent using the base encoding type (`application/x-www-form-urlencoded`) is handled via the `data=` parameter.
# * To POST a message body directly, pass string or bytes type data to this parameter.
# * To pass form-encoded data, pass dict or tuple type data to this parameter.
# * To pass JSON data, the `json=` parameter is helpful. This is ultimately just an alias for `data=json.dumps(payload)`.
# * To POST data out of a file, using multipart encoding (e.g. `enctype='multipart/form-data'`), use `files=`. The `Content-Type` and additional headers included in the follow code sample are optional:
# 
#     ```
#     files = {'file': ('report.xls', open('report.xls', 'rb'), 'application/vnd.ms-excel', {})}
#     r = requests.post(url, files=files)
#     ```
#     
#   How do you process this kind of input on the server side? You probably don't want to; you want your server software to handle it. But [here's a good answer showcasing the receipt format](https://stackoverflow.com/a/28380690/1993206).
# 
# ### Cookies
# * Cookies can be passed using `cookies`:
# 
#   ```python
#   r = requests.get(url, cookies={"field": "value"})
#   ```
# 
# * They are available in the `cookies` field on the response, e.g. `r.cookies`.
