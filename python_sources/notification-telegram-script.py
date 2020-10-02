#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
This simple script defines the class TNotifier
who let send messages through Telegram API.

Please read https://core.telegram.org/bots/api.

To send message through Telegram, you need to use
an access token, and a group chat id.

use case:
- know when your committing gets finished
- know the progressive of your code
- know the caught errors
"""

__license__ = "WTFPL"

import requests

URL = "https://api.telegram.org/bot{token}/{method}"

class TNotifier(object):
    def __init__(self, token:str, gid:int, username:str):
        """Init the object
        
        :param token: access token
        :param gid: group chat id
        :param username: your Telegram username
        """
        super(TNotifier, self).__init__()
        self.token = token
        self.gid = gid
        self.username = username
        if not self.username.startswith("@"):
            self.username = "@" + self.username

    def report(self, e:Exception, title:str=""):
        """Report an error.
        
        :param e: catched exception
        :param title: adjunct message
        """
        if not title:
            title = "catched"
        self.simple_notify(
            "{title}: ({cls}) {message}".format(
                title=title,
                cls=e.__class__.__name__,
                message=str(e)),
            mention=True)
        raise e

    def simple_notify(self, text:str, mention:bool=False):
        """Send a simple notification.
        
        :param text: text of message
        :param mention: set whether have to mention
        """
        if mention:
            report = "{username}, {text}"
        else:
            report = "{text}"
        self._send_text(
            self.gid,
            report.format(
                username=self.username,
                text=text))

    def _send_text(self, gid:int, text:str):
        """Send text by Telegram.
        
        :param gid: group chat id
        :param text: text
        """
        return self._make_request(
            "sendMessage",
            dict(
                chat_id=self.gid,
                text=text))

    def _make_request(self, method:str, params:dict=[]):
        """Make a request.
        
        :param method: method name
        :param params: requested URL params
        :return dict: returns Telegram answer
        """
        # Create the URL
        url = URL.format(token=self.token, method=method)
        web = requests.get(
            url,
            params=params)
        if not web.ok:
            print(web.status_code, web.reason, web.json())
            return None
        return web.json()