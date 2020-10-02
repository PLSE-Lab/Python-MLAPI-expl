#script to get the dataset

from binance.client import Client
client = Client("", "")

def getcandles():
    candles = client.get_historical_klines(
                coin, Client.KLINE_INTERVAL_1DAY, "16 Jul, 2017", "5 May, 2018")
                
    opentime=[]
    lopen=[]
    lhigh=[]
    llow=[]
    lclose=[]
    lvol=[]
    closetime=[]

    for candle in candles:
        opentime.append(candle[0])
        lopen.append(candle[1])
        lhigh.append(candle[2])
        llow.append(candle[3])
        lclose.append(candle[4])
        lvol.append(candle[5])
        closetime.append(candle[6])
        
    lopen=np.array(lopen).astype(np.float)
    lhigh=np.array(lhigh).astype(np.float)
    llow=np.array(llow).astype(np.float)
    lclose=np.array(lclose).astype(np.float)
    lvol=np.array(lvol).astype(np.float)
    return lopen, lhigh, llow, lclose, lvol, closetime
   
def getCoins():
    products = client.get_products()
    coins = []
    for x in range (0, len(products['data'])):
        coin = products['data'][x]['symbol']
        if (coin[len(coin) - 3: len(coin)] == 'BTC'):
            coins.append(str(products['data'][x]['symbol']))
    return coins
        
def getAllDatabases():
    coins = getCoins()
    periods = ['Day', 'hour', 'thirtyMin']
    start  = 'Jul, 2017' 
    end = 'May, 2018'

    for coin in coins:
        print(coin)
        for period in periods:
            lopen, lhigh, llow, lclose, lvol, closetime = self.getCandles(coin, period)
            hp.saveDatabase(lopen, lhigh, llow, lclose, lvol, closetime, coin, period, start, end)
    
def saveDatabase(lopen, lhigh, llow, lclose, lvol, closetime, coin, period, start, end):
    thefile = open('databases/' + str(coin) + '-' + str(period) + '-' +
                   str(start) + '-' + str(end)+'.csv', 'w')
    thefile.write("Date,Open,High,Low,Close,Adj  Close,Volume \n")
    for item in range(0, len(lopen)):
        thefile.write(
            str(closetime[item]) + ',' + str(lopen[item]) +
            ',' + str(lhigh[item]) + ','
            + str(llow[item]) + ',' + str(lclose[item]) + ',' + str(lclose[item]) +','+ str(lvol[item]) + '\n')
            
