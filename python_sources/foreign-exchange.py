#!/usr/bin/env python
# coding: utf-8

# # Over All flow:
# 
# 1. Take country name from user
# 2. Check with all the values of synonyms and return the key where value matched from **synonyms**
# 3. Convert that value to currency code through **code_to_fullname**
# 4. Put that currencty code to **currency_to_nepali**(default value = 1 but can be changed)
# 5. Function **currency_to_nepali** return the amount in NRP

# # Foreign exchange rate
# 
# * since the INR to NPR is a fixed value , we will take it

# In[ ]:


get_ipython().system('pip install forex-python')


# In[ ]:


from forex_python.converter import CurrencyRates, CurrencyCodes


# In[ ]:


c = CurrencyRates()


# In[ ]:


# currency code ---> full name
code_to_fullname = { 
    "EUR" : "Euro Member Countries" ,
    "IDR" : "Indonesia Rupiah" ,
    "BGN" : "Bulgaria Lev" ,
    "ILS" : "Israel Shekel" ,
    "GBP" : "United Kingdom Pound", 
    "DKK" : "Denmark Krone" ,
    "CAD" : "Canada Dollar" ,
    "JPY" : "Japan Yen" ,
    "HUF" : "Hungary Forint", 
    "RON" : "Romania New Leu", 
    "MYR" : "Malaysia Ringgit", 
    "SEK" : "Sweden Krona" ,
    "SGD" : "Singapore Dollar", 
    "HKD" : "Hong Kong Dollar" ,
    "AUD" : "Australia Dollar" ,
    "CHF" : "Switzerland Franc" ,
    "KRW" : "Korea (South) Won" ,
    "CNY" : "China Yuan Renminbi", 
    "TRY" : "Turkey Lira" ,
    "HRK" : "Croatia Kuna" ,
    "NZD" : "New Zealand Dollar", 
    "THB" : "Thailand Baht" ,
    "USD" : "United States Dollar", 
    "NOK" : "Norway Krone" ,
    "RUB" : "Russia Ruble" ,
    "INR" : "India Rupee" ,
    "MXN" : "Mexico Peso" ,
    "CZK" : "Czech Republic Koruna", 
    "BRL" : "Brazil Real" ,
    "PLN" : "Poland Zloty" ,
    "PHP" : "Philippines Peso", 
    "ZAR" : "South Africa Rand"
}


# In[ ]:


synonyms = {

'Euro Member Countries' : ['Euro','Europe', 'EUR'], 
'Indonesia Rupiah': ['Indonesia', 'IDR'], 
'Bulgaria Lev': ['Bulgaria', 'BGN'], 
'Israel Shekel' : ['Israel', 'ILS'], 
'United Kingdom Pound' : ['GBP','United Kingdom','UK', 'U.K.','U.K','England', 'Scotland', 'Wales', 'Northern Ireland','Britain', ' Pound'], 
'Denmark Krone' : ['DKK','Denmark'], 
'Canada Dollar' : ['CAD','Canada'],
'Japan Yen': ['JPY','Japan'],
'Hungary Forint': ['HUF','Hungary'],
 'Romania New Leu' : ['RON','Romania'], 
 'Malaysia Ringgit' : ['MYR','Malaysia'], 
 'Sweden Krona' : ['SEK','Sweden'], 
 'Singapore Dollar' : ['SGD','Singapore'], 
 'Hong Kong Dollar' : ['HKD','Hong Kong', 'HongKong'], 
 'Australia Dollar' : ['AUD','Australia'], 
 'Switzerland Franc' : ['CHF','Switzerland'], 
 'Korea (South) Won' : ['KRW','South korea','SouthKorea'], 
 'China Yuan Renminbi' : ['CNY','China'], 
 'Turkey Lira' : ['TRY','Turkey'], 
 'Croatia Kuna' : ['HRK','Croatia'], 
 'New Zealand Dollar' : ['NZD','New Zealand','NewZealand'], 
 'Thailand Baht' : ['THB','Thailand'], 
 'United States Dollar' : ['USD','United States','USA','US','U.S''U.S.','U.S.A','U.S.A.', 'America'], 
 'Norway Krone' : ['NOK','Norway'], 
 'Russia Ruble' : ['RUB','Russia'], 
 'India Rupee' : ['INR','India'], 
 'Mexico Peso' : ['MXN','Mexico'], 
 'Czech Republic Koruna' : ['CZK','Czech Republic','CzechRepublic'], 
 'Brazil Real' : ['BRL','Brazil'], 
 'Poland Zloty' : ['PLN','Poland'], 
 'Philippines Peso' : ['PHP','Philippines'], 
 'South Africa Rand' : ['ZAR','South Africa', 'SouthAfrica'] 

 }


# In[ ]:


#country name ---> code
fullname_to_code = {}
for key, value in code_to_fullname.items():
    fullname_to_code[value] = key


# In[ ]:


fullname_to_code.keys()


# # Country name to NPR Currency

# In[ ]:


def currency_to_nepali(currency_code, value = 1):
    c = CurrencyRates()
    nepali_currency = c.convert(currency_code, 'INR', value)*1.6001
    return nepali_currency


# In[ ]:


def country_name_to_currency(currency_name, currency_value=1):
    
    for key, value in synonyms.items():
        for val1 in value:
            #print(val1)
            if val1.lower() == currency_name.lower():
                currency_code = key 
                break;
    try:
        currency_code = fullname_to_code[currency_code]
        currency_in_nepali = currency_to_nepali(currency_code, currency_value)
        return currency_in_nepali
    except:
        print("An exception occurred! Write country name properly") 


# In[ ]:


country_name = 'Chna'
currency = country_name_to_currency(country_name)
print(currency)


# In[ ]:


country_name = 'USD'
currency = country_name_to_currency(country_name,3)
print(currency)

