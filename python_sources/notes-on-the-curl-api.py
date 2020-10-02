#!/usr/bin/env python
# coding: utf-8

# ## Notes on the curl API
# 
# `curl` works with a lot of different protocols, these notes only pertain to the part that I care about: HTTP and HTTPS. These notes based on https://curl.haxx.se/docs/manual.html.
# 
# ### GET
# * Download a page to a specific output filename: `curl -o thatpage.html http://www.netscape.com/`
# * Can also dump headers (to a separate file) using `--dump-header`: `curl --dump-header headers.txt curl.haxx.se`.
# * Can also dump cookies using the `-c` flag.
# 
# ### Authorization
# * There are plenty of HTTP authorization schemes, e.g. OAuth2. The simplest, HTTP Basic Authorization, is supported via the following: `curl -u name:passwd http://machine.domain/full/path/to/file`
# 
# ### Headers
# * The `-H` flag. E.g. `curl -X POST -H "Content-Type: application/octet-stream"`
# 
# ### PUT and POST
# * PUT from `stdin`: `curl -X PUT -T - http://www.upload.com/target_url`. E.g.:
#   * `echo Foo | curl -X PUT -T - http://www.upload.com/`
#   * `cat - file.txt | curl -X PUT -T - http://www.upload.com/`
#   
# 
# * POST data: `curl -X POST -d "name=Rafael%20Sagula&phone=3320780" http://www.where.com/guest.cgi`. Data must be URL-encoded. This is a `application/x-www-form-urlencoded` POST.
# * To POST binary data using the `application/x-www-form-urlencoded` scheme, use the `--data-binary` flag instead.
# * These parameters can read data from filenames specified by a leading `@` character. E.g. here's a command I've actually run: `curl -X POST http://127.0.0.1:5000/predict --data-binary "@/Users/alex/Desktop/test_image_label_map.png"`
# 
# 
# * To send form-fill data, use the `-F` flag: `curl -F "coolfiles=@fil1.gif;type=image/gif,fil2.txt,fil3.html" http://www.post.com/postit.cgi`.
# * Note the use of `;type=image/gif` to specify the content type as well, inline.
# * The names of the fields are field names in the HTTP source. E.g. `curl -F "file=@cooltext.txt" -F "filedescription=Cool http://www.post.com/postit.cgi` posts to the `file` and `filedescription` form fields on the page.
# * If you need to use the `@` character you cannot, using `-F`. Use `--form-string` instead, in this case.
# 
# 
# ### Cookies
# * Cookies can be saved to a local file using the `-c` flag. E.g. `curl -c cookies.txt www.example.com`.
# * Cookies can be specified via the `-b` flag. E.g. `curl -b "name=Daniel" www.sillypage.com`. Or to specify from a file: `curl -b cookies.txt www.sillypage.com`.
# * You can read and write cookies simultaneously using `curl -b cookies.txt -c cookies.txt www.example.com`. This may be useful for maintaining certain session data.
# 
# 
# ### Session resumption
# * `curl` supports resuming a stopped or killed download session using the `-C` flag: `curl -C - -o file ftp://ftp.server.com/path/file`.
# 
# ### Netrc
# * `curl` supports the `.netrc` file (on Linux) IFF you specify `--netrc`. This is otherwise disabled by default.
