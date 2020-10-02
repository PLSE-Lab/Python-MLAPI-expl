import http.client
import json

#Sample language itc
#Can be found by inspecting http response of google input tools page
malayalam = 'ml-t-i0-und'
hindi = 'hi-t-i0-und'
telugu = 'te-t-i0-und'

def request(input, itc):
    conn = http.client.HTTPSConnection('inputtools.google.com')
    conn.request('GET', '/request?text=' + input + '&itc=' + itc + '&num=1&cp=0&cs=1&ie=utf-8&oe=utf-8&app=test')
    res = conn.getresponse()
    return res

def driver(input, itc):
    output = ''
    if ' ' in input:
        input = input.split(' ')
        for i in input:
            res = request(input = i, itc = itc)
            res = res.read()
            if i==0:
                output = str(res, encoding = 'utf-8')[14+4+len(i):-31]
            else:
                output = output + ' ' + str(res, encoding = 'utf-8')[14+4+len(i):-31]
    else:
        res = request(input = input, itc = itc)
        res = res.read()
        output = str(res, encoding = 'utf-8')[14+4+len(input):-31]
    print(output)
    
driver('saloni kalra', malayalam)
driver('saloni kalra', hindi)
driver('saloni kalra', telugu)