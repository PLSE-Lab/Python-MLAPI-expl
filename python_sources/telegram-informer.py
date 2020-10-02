# %% [code]
import requests
from requests.exceptions import ConnectionError


# %% [code]
CONNECT_ERROR_MESSAGE='Error connecting with Telegram API - Switch on "Internet" option in Settings'
RESP_400_MESSAGE='400 Error - Incorrect chat_id'
RESP_401_MESSAGE='401 Error - Incorrect bot token'
MAX_100_UPDATES=100

# %% [code]
def to_telegram(mess, bot_token, chat_id):
    """Send text message to Telegram user or channel.
       Instructions to create bot https://core.telegram.org/bots
    Args:
        mess (str): Message.
        bot_token (str): 'YYYYYYYYY:AAGENOh2txET_prrkcSSGccbqXXXXXXXXXX' 
                         - token received from @BotFather
        chat_id (int): 69111111 - chat ID, you may need @userinfobot or @ShowJsonBot. 
                       Negative value for channels.

    Returns:
        str: "OK" or error message.
    """
    
    address="https://api.telegram.org/bot{}/sendMessage".format(bot_token)
    data = {'chat_id': chat_id, 'text': mess}
    try:
        resp = requests.post(address, data=data)
        if not resp.ok:
            if resp.status_code==400:
                return RESP_400_MESSAGE
            elif resp.status_code==401:
                return RESP_401_MESSAGE
            return resp.reason
    except ConnectionError as e:
        return CONNECT_ERROR_MESSAGE
    return "OK"


# %% [code]

class DashBot:
    """Exchange text messages with Telegram user or channel."""
    def __init__(self, token, chat_id):
        """
        Args:
            bot_token (str): '235787775:AAGENOh2txET_prrkcSSGccbqXXXXXXXXXX' 
                             - token received from @BotFather
            chat_id (int): 69111111 - chat ID, you may need @userinfobot or @ShowJsonBot. 
                           Negative value for channels.
        """
        
        self._chat_id=chat_id
        self._message_text=''
        self._internet_OFF=False
        self._token=token
        self._chat_id=chat_id
        self._next_update_id=-1
        self._reason=""

        #forget old incoming messages
        self.get_last_update()

    def error_reason(self):
        return self._reason
    
    def __call_api(self, api_type, data):
        address="https://api.telegram.org/bot{}/{}".format(self._token, api_type)
        try:
            resp = requests.post(address, data=data)
            self._reason=resp.reason
            if resp.ok:
                return resp.json()['result']
        except ConnectionError as e:
            self._reason = CONNECT_ERROR_MESSAGE
        return None
            
    def send_message (self, text):
        """Send message and save its message_id to later use in add_text()"""
        if self._internet_OFF:
            return
        data = {'chat_id': self._chat_id, 'text': text}
        res = self.__call_api('sendMessage', data)
        if res is not None:
            self._last_mess_id=res['message_id']
            self._message_text=text
        
    #update last message
    def add_text (self, text):
        """Update existing message by saved message_id"""
        if self._internet_OFF:
            return
        if self._last_mess_id==None:
            self.send_message(text)
        else:
            self._message_text+='\n'+text
            data = {'chat_id': self._chat_id, 'message_id':self._last_mess_id, 'text': self._message_text}
            self.__call_api('editMessageText', data)

    def get_last_update (self):
        """Discard all incoming messages and return last"""
        if self._internet_OFF:
            return
        res = ''
        #read all updates and return last
        while True:
            data = {'chat_id': self._chat_id, 'offset': self._next_update_id, 'limit': MAX_100_UPDATES, 'allowed_updates': '["message"]'}
            resp = self.__call_api('getUpdates', data)
            try:
                self._next_update_id=res[-1]['update_id']+1
                res = resp[-1]['message']['text']
                if len(resp) < MAX_100_UPDATES:
                    break
            except:
                break
        return res
            
