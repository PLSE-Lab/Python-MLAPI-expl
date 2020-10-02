#!/usr/bin/env python
# coding: utf-8

# ## Notes on cookies
# 
# Notes on cookies, taken directly from the RFC: https://www.ietf.org/rfc/rfc2109.txt. Just a snippet.
# 
# * Cookies provide sessionization. They are the preferred method for persisting data about a user as they browse your website, like login information and other things.
# * A server sends a `Set-Cookie` in the response header which includes the cookie payload information. The client is supposed to send that cookie payload with every request it makes to the server thereafter.
# * As the session evolves, the server may update the cookie by sending another `Set-Cookie` response.
# 
# 
# * A `Set-Cookie` header has a number of fields. A `Max-Age` field sets a TTL; a `Name` field provides a...name, a `Value` field is an opaque (to the user agent) data payload that the server can read from and use.
# * `Cache-control` is an important field here because it controls caching of the page. E.g. a server cannot expect to be able to use its own front page as the start of a session, if that front page is sometimes cached. The server may request that the client verify that its copy is still current. This is a faster network roundtrip: an `HTTP 304 Not Modified`. It may even require a redownload every time by setting a `Max-Age=0`.
# * You may close a session by sending a `Max-Age=0`.
# 
# 
# * A user agent may request a cookie by sending a cookie request, which is different from a `Set-Cookie` response. How the server handles this is up to the server (send a new cookie, cache and send the old cookie, etc.).
# 
# 
# * By default a cookie is only valid for the current subdomain. If you pass a `Domain` value, you may specify that a cookie is valid for the enture current domain or subdomain of the entire current domain. This means that e.g. `a.foo.com` can set a cookie which appears on `foo.com` as well.
# * This is the reason why you need to keep official resources and user-definable "unsafe" resources on separate domains entirely. Users in a playground can set cookies on your actual domain by setting the right `Domain` value, allowing them to interact with the user session and server response on your primary domain.
# * This is the `helix.quiltdata.com` story about giving them a subdomain of ours so they don't have to mint and manage a new AWS HTTPS certificate.
# * To test this out try out: https://scripts.cmbuckley.co.uk/cookies.php
