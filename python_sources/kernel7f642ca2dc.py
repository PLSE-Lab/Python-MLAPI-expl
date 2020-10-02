#!pip install -q matplotlib-venn
import websocket
ws = websocket.WebSocketApp('wss://api.bitfinex.com/ws/2')

ws.on_open = lambda self: self.send('{ "event": "subscribe",  "channel": "candles",  "key": "trade:1m:tBTCUSD" }')

ws.on_message = lambda self, evt:  print (evt)
