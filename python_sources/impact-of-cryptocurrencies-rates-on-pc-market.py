#!/usr/bin/env python
# coding: utf-8

# ## Impact of cryptocurrencies rates on PC parts prices
# ### University project data analysis results
# >Kamil Raczycki, may 2018
# 
# I'd like to share my results of this analysis with some Python code that I've used to semi-automatically scrap data from 3 comparator engines: PC Part Picker, Price Spy and Geizhals. 
# 
# Charts aren't interactive because I copied them from Tableau where I made these analytics after creating a cube.

# *This notebook is not completed yet!*
# ### TODO:
# * Add charts about Cryptocurrency impact on CPUs
# * Add charts about Cryptocurrency impact on RAMs
# * Add charts about GPUs prices differences between regions
# * Add charts about GPUs prices differences between manufacturers and models
# * Add charts about CPUs prices differences between regions
# * Add charts about CPUs prices differences between manufacturers and models
# * Add charts about RAM prices differences between regions
# * Add charts about RAM prices differences between manufacturers and models
# * Add comments to each chart with analysis and conclusions
# 

# ## Top 10 most popular GPUs used for mining
# -----
# ![Top 10 most popular GPUs used for mining](https://i.imgur.com/yjWpNTH.png)

# ## GPUs clustered by amount of VRAM
# -----
# ![GPUs clustered by amount of VRAM](https://i.imgur.com/eoknlcF.png)

# ## GPUs clustered by VRAM type
# -----
# ![GPUs clustered by VRAM type](https://i.imgur.com/aeD5bbg.png)

# ### Currency Scrapping
# -----
# ```python
# ### Currency scrapper
# # -*- coding: utf8 -*-
# import os
# import urllib
# from unicodecsv import DictReader, DictWriter
# from datetime import datetime
# from json import loads
# from time import sleep
# 
# from BeautifulSoup import BeautifulSoup
# 
# def save_csv_file(file_name, data):
#     if len(data) > 0:
#         with open(file_name, 'w+b') as csvfile:
#             fieldnames = data[0].keys()
#             writer = DictWriter(csvfile, fieldnames=fieldnames)
#             writer.writeheader()
#             writer.writerows(data)
# 
# def load_currency_rate_from_html(file_name, currency):
#     parsed_data = []
#     with open(file_name, 'r') as htmlfile:
#         soup = BeautifulSoup(htmlfile)
#         body = soup.find('tbody')
#         for el in body.findAll('tr'):
#             date_str = el.find('td', {'class': 'historical-rates--table--date ng-binding'}).text
#             rate_str = el.find('td', {'class': 'historical-rates--table--rate ng-binding'}).text
#             d = datetime.strptime(date_str, '%B %d, %Y')
#             date_parsed = d.strftime('%Y-%m-%d')
#             rate_parsed = float(rate_str)
#             temp_dict = {
#                 'date': date_parsed,
#                 'to_rate': rate_parsed,
#                 'from': 'USD',
#                 'to': currency,
#                 'from_rate': 1.0
#             }
#             parsed_data.append(temp_dict)
#         return parsed_data
# 
# files = ['USD_AUD_au', 'USD_CAD_ca', 'USD_EUR_be_de_es_fr_ie_it', 'USD_GBP_uk', 'USD_IND_in', 'USD_NZD_nz']
# currencies = ['AUD', 'CAD', 'EUR', 'GBP', 'IND', 'NZD']
# for f, c in zip(files, currencies):
#     data = load_currency_rate_from_html('Data\\currencies\\{0}.html'.format(f), c)
#     file_name = 'USD_{0}'.format(c)
#     save_csv_file('Data\\currencies\\{0}.csv'.format(file_name),data)
# ```

# ### PC Part Picker
# -----
# ```python
# ### PC Part Picker [it's ulgy, I know, but I didn't have time to clean it :( 
# # -*- coding: utf8 -*-
# import os
# import urllib
# from unicodecsv import DictReader, DictWriter
# from datetime import datetime
# from json import loads
# from time import sleep
# 
# import requests
# from BeautifulSoup import BeautifulSoup
# from PCPartPicker_API import PCPartPicker as pcpartpicker
# 
# 
# def save_csv_file(file_name, data):
#     if len(data) > 0:
#         with open(file_name, 'w+b') as csvfile:
#             fieldnames = data[0].keys()
#             writer = DictWriter(csvfile, fieldnames=fieldnames)
#             writer.writeheader()
#             writer.writerows(data)
# 
# def load_ids_from_csv(file_name):
#     ids = []
#     with open(file_name, 'r') as csvfile:
#         reader = DictReader(csvfile)
#         ids = [row['prodId'] for row in reader]
#     return ids
# 
# def parse_one_gpu(gpu_id, region='us', get_memory_type=False):
#     prefix = '' if region == 'us' else '{0}.'.format(region)
#     url = 'http://{0}pcpartpicker.com/product/{1}'.format(prefix, gpu_id)
#     params = {
#         'history_days': 730
#     }
#     scrapped = False
#     amount = 1
#     while not scrapped:
#         try:
#             connected = False
#             tries = 10
#             while not connected:
#                 try:
#                     sleep(0.5)
#                     resp = requests.get(url=url, params=params)
#                     connected = True
#                 except:
#                     print('connection problem, trying again...')
#                     tries -= 1
#                     if tries < 0:
#                         raise
#             soup = BeautifulSoup(resp.text)
#             # find memory type
#             actual_idx = -1
#             mem_type = None
#             if get_memory_type:
#                 specs = soup.find('div', {'class': 'specs block'})
#                 for i, line in enumerate(specs):
#                     if i == actual_idx:
#                         mem_type = line.strip()
#                         break
#                     elif line.string == 'Memory Type':
#                         actual_idx = i+1
#             # find price data
#             raw_data = []
#             parsed_data = []
#             scripts = soup.findAll('script')
#             for script in scripts:
#                 if 'phistmulti' in script.text:
#                     data = script.prettify().split('\n')
#                     for line in data:
#                         if 'phistmulti' in line:
#                             idx = line.index('[')
#                             price_hist = line[idx:-1]
#                             raw_data = loads(price_hist)
#                             break
#             for mer_row in raw_data:
#                 merchant = mer_row['label'].replace('&amp;', ' ')
#                 for value in mer_row['data']:
#                     price = value[1]
#                     if price is None:
#                         continue
#                     price = float(price) / 100
#                     utc_time = datetime.utcfromtimestamp(value[0]/1000)
#                     temp_dict = {
#                         'prodId': gpu_id,
#                         'date': utc_time.strftime("%Y-%m-%d"),
#                         'price': price,
#                         'region': region,
#                         'merchant': merchant
#                     }
#                     parsed_data.append(temp_dict)
#             scrapped = True
#         except:
#             print('CAPTCHA problem, trying again... {0}'.format(amount))
#             amount += 1
#             if amount > 100:
#                 raise
#             sleep(5)
#     return parsed_data, mem_type
# 
# def parse_one_cpu(gpu_id, region='us', get_socket_type=False):
#     prefix = '' if region == 'us' else '{0}.'.format(region)
#     url = 'http://{0}pcpartpicker.com/product/{1}'.format(prefix, gpu_id)
#     params = {
#         'history_days': 730
#     }
#     scrapped = False
#     amount = 1
#     while not scrapped:
#         try:
#             connected = False
#             tries = 10
#             while not connected:
#                 try:
#                     sleep(0.5)
#                     resp = requests.get(url=url, params=params)
#                     connected = True
#                 except:
#                     print('connection problem, trying again...')
#                     tries -= 1
#                     if tries < 0:
#                         raise
#             soup = BeautifulSoup(resp.text)
#             # find memory type
#             actual_idx = -1
#             mem_type = None
#             if get_socket_type:
#                 specs = soup.find('div', {'class': 'specs block'})
#                 for i, line in enumerate(specs):
#                     if i == actual_idx:
#                         mem_type = line.strip()
#                         break
#                     elif line.string == 'Socket':
#                         actual_idx = i+1
#             # find price data
#             raw_data = []
#             parsed_data = []
#             scripts = soup.findAll('script')
#             for script in scripts:
#                 if 'phistmulti' in script.text:
#                     data = script.prettify().split('\n')
#                     for line in data:
#                         if 'phistmulti' in line:
#                             idx = line.index('[')
#                             price_hist = line[idx:-1]
#                             raw_data = loads(price_hist)
#                             break
#             for mer_row in raw_data:
#                 merchant = mer_row['label'].replace('&amp;', ' ')
#                 for value in mer_row['data']:
#                     price = value[1]
#                     if price is None:
#                         continue
#                     price = float(price) / 100
#                     utc_time = datetime.utcfromtimestamp(value[0]/1000)
#                     temp_dict = {
#                         'prodId': gpu_id,
#                         'date': utc_time.strftime("%Y-%m-%d"),
#                         'price': price,
#                         'region': region,
#                         'merchant': merchant
#                     }
#                     parsed_data.append(temp_dict)
#             scrapped = True
#         except:
#             print('CAPTCHA problem, trying again... {0}'.format(amount))
#             amount += 1
#             if amount > 100:
#                 raise
#             sleep(5)
#     return parsed_data, mem_type
# 
# def parse_one_ram(gpu_id, region='us', get_socket_type=False):
#     prefix = '' if region == 'us' else '{0}.'.format(region)
#     url = 'http://{0}pcpartpicker.com/product/{1}'.format(prefix, gpu_id)
#     params = {
#         'history_days': 730
#     }
#     scrapped = False
#     amount = 1
#     while not scrapped:
#         try:
#             connected = False
#             tries = 10
#             while not connected:
#                 try:
#                     sleep(0.5)
#                     resp = requests.get(url=url, params=params)
#                     connected = True
#                 except:
#                     print('connection problem, trying again...')
#                     tries -= 1
#                     if tries < 0:
#                         raise
#             soup = BeautifulSoup(resp.text)
#             # find memory type
#             actual_idx = -1
#             mem_type = None
#             if False and get_socket_type:
#                 specs = soup.find('div', {'class': 'specs block'})
#                 for i, line in enumerate(specs):
#                     if i == actual_idx:
#                         mem_type = line.strip()
#                         break
#                     elif line.string == 'Socket':
#                         actual_idx = i+1
#             # find price data
#             raw_data = []
#             parsed_data = []
#             scripts = soup.findAll('script')
#             for script in scripts:
#                 if 'phistmulti' in script.text:
#                     data = script.prettify().split('\n')
#                     for line in data:
#                         if 'phistmulti' in line:
#                             idx = line.index('[')
#                             price_hist = line[idx:-1]
#                             raw_data = loads(price_hist)
#                             break
#             for mer_row in raw_data:
#                 merchant = mer_row['label'].replace('&amp;', ' ')
#                 for value in mer_row['data']:
#                     price = value[1]
#                     if price is None:
#                         continue
#                     price = float(price) / 100
#                     utc_time = datetime.utcfromtimestamp(value[0]/1000)
#                     temp_dict = {
#                         'prodId': gpu_id,
#                         'date': utc_time.strftime("%Y-%m-%d"),
#                         'price': price,
#                         'region': region,
#                         'merchant': merchant
#                     }
#                     parsed_data.append(temp_dict)
#             scrapped = True
#         except:
#             print('CAPTCHA problem, trying again... {0}'.format(amount))
#             amount += 1
#             if amount > 100:
#                 raise
#             sleep(5)
#     return parsed_data, mem_type
# 
# def load_gpu_data_from_api(skip=[]):
#     temp_gpu_data = []
#     gpu_prices = []
#     gpu_data = []
# 
#     lists = pcpartpicker.lists.total_pages("video-card")
#     print(lists)
#     gpu_list = []
#     for l in range(lists):
#         gpu_list += pcpartpicker.lists.get_list("video-card", l+1)
#     for gpu in gpu_list:
#         temp_dict = {
#             'prodId': gpu['id'],
#             'name': gpu['name'],
#             'processor': gpu['chipset'],
#             'memorycapacity': gpu['memory'],
#             'memorytype': None
#         }
#         temp_gpu_data.append(temp_dict)
#     print(len(temp_gpu_data))
#     regions = ["au", "be", "ca", "de", "es", "fr", "in", "ie", "it", "nz", "uk"] # "us"
#     for idx, gpu in enumerate(temp_gpu_data):
#         if idx+1 in skip:
#             print(idx+1, 'skipping')
#             continue 
#         temp_list, memory_type = parse_one_gpu(gpu['prodId'], get_memory_type=True)
#         gpu['memorytype'] = memory_type
#         gpu_data.append(gpu)
#         print(idx+1, gpu)
#         print('us', len(temp_list))
#         gpu_prices.extend(temp_list)
#         for region in regions:
#             temp_list, _ = parse_one_gpu(gpu['prodId'], region=region)
#             print(region, len(temp_list))
#             gpu_prices.extend(temp_list)
#         if len(gpu_data) == 5:
#             print(idx+1, 'saving data')
#             save_data(gpu_data, gpu_prices, 'GPU')
#             gpu_prices = []
#             gpu_data = []
#     save_data(gpu_data, gpu_prices, 'GPU')
# 
# def load_cpu_data_from_api(skip=[]):
#     temp_cpu_data = []
#     cpu_prices = []
#     cpu_data = []
# 
#     lists = pcpartpicker.lists.total_pages("cpu")
#     print(lists)
#     cpu_list = []
#     for l in range(lists):
#         cpu_list += pcpartpicker.lists.get_list("cpu", l+1)
#     for cpu in cpu_list:
#         temp_dict = {
#             'prodId': cpu['id'],
#             'name': cpu['name'],
#             'cores': cpu['cores'],
#             'type': cpu['name'],
#             'socket': None
#         }
#         temp_cpu_data.append(temp_dict)
#     print(len(temp_cpu_data))
#     regions = ["au", "be", "ca", "de", "es", "fr", "in", "ie", "it", "nz", "uk"] # "us"
#     for idx, cpu in enumerate(temp_cpu_data):
#         if idx+1 in skip:
#             print(idx+1, 'skipping')
#             continue 
#         temp_list, socket_type = parse_one_cpu(cpu['prodId'], get_socket_type=True)
#         cpu['socket'] = socket_type
#         cpu_data.append(cpu)
#         print(idx+1, cpu)
#         print('us', len(temp_list))
#         cpu_prices.extend(temp_list)
#         for region in regions:
#             temp_list, _ = parse_one_cpu(cpu['prodId'], region=region)
#             print(region, len(temp_list))
#             cpu_prices.extend(temp_list)
#         if len(cpu_data) == 5:
#             print(idx+1, 'saving data')
#             save_data(cpu_data, cpu_prices, 'CPU')
#             cpu_prices = []
#             cpu_data = []
#     save_data(cpu_data, cpu_prices, 'CPU')
# 
# def load_ram_data_from_api(skip=[]):
#     temp_ram_data = []
#     ram_prices = []
#     ram_data = []
# 
#     lists = pcpartpicker.lists.total_pages("memory")
#     print(lists)
#     ram_list = []
#     for l in range(lists):
#         ram_list += pcpartpicker.lists.get_list("memory", l+1)
#     for ram in ram_list:
#         speed = ram['speed'].split('-')
#         temp_dict = {
#             'prodId': ram['id'],
#             'name': ram['name'],
#             'capacity': ram['size'],
#             'speed': speed[1],
#             'type': speed[0]
#         }
#         temp_ram_data.append(temp_dict)
#     print(len(temp_ram_data))
#     regions = ["au", "be", "ca", "de", "es", "fr", "in", "ie", "it", "nz", "uk"] # "us"
#     for idx, ram in enumerate(temp_ram_data):
#         if idx+1 in skip:
#             print(idx+1, 'skipping')
#             continue 
#         temp_list, _ = parse_one_ram(ram['prodId'], get_socket_type=True)
#         # ram['socket'] = socket_type
#         ram_data.append(ram)
#         print(idx+1, ram)
#         print('us', len(temp_list))
#         ram_prices.extend(temp_list)
#         for region in regions:
#             temp_list, _ = parse_one_ram(ram['prodId'], region=region)
#             print(region, len(temp_list))
#             ram_prices.extend(temp_list)
#         if len(ram_data) == 5:
#             print(idx+1, 'saving data')
#             save_data(ram_data, ram_prices, 'RAM')
#             ram_prices = []
#             ram_data = []
#     save_data(ram_data, ram_prices, 'RAM')
# 
# def concat_csv_prices_files(directory, folder, component):
#     files = os.listdir(directory)
#     data = []
#     counter = 0
#     for file_name in files:
#         if folder == 'pricespy' and component == 'GPU' and not 'uk' in file_name:
#             continue
#         temp_counter = 0
#         with open('{0}\{1}'.format(directory,file_name), 'r') as csvfile:
#             reader = DictReader(csvfile)
#             for row in reader:
#                 data.append(row)
#                 temp_counter += 1
#         counter += temp_counter
#         print(file_name, temp_counter, counter)
#     print("saving data")
#     save_csv_file('Data\\{0}\\{1}_Prices_merged\\{2}_merged_{3}.csv'.format(folder, component, datetime.now().strftime("%Y%m%d_%H%M%S_%f"), counter), data)
# 
# def save_data(data, prices, component):
#     save_csv_file('Data\\pcpartpicker\\{0}_Types\\{1}.csv'.format(component, datetime.now().strftime("%Y%m%d_%H%M%S_%f")), data)
#     save_csv_file('Data\\pcpartpicker\\{0}_Prices\\{1}.csv'.format(component, datetime.now().strftime("%Y%m%d_%H%M%S_%f")), prices)
# 
# load_gpu_data_from_api()
# load_cpu_data_from_api()
# load_ram_data_from_api()
# ```

# ### Geizhals
# -----
# ```python
# ### Geizhals
# # -*- coding: utf8 -*-
# import urllib
# from csv import DictReader, DictWriter
# from datetime import datetime
# from json import loads
# from time import sleep
# 
# import requests
# from BeautifulSoup import BeautifulSoup
# from PCPartPicker_API import PCPartPicker as pcpartpicker
# 
# def save_csv_file(file_name, data):
#     with open(file_name, 'w+b') as csvfile:
#         fieldnames = data[0].keys()
#         writer = DictWriter(csvfile, fieldnames=fieldnames)
#         writer.writeheader()
#         writer.writerows(data)
# 
# def load_links_from_html(file_name):
#     links = []
#     with open(file_name, 'r') as htmlfile:
#         soup = BeautifulSoup(htmlfile)
#         for el in soup.findAll('a', {'class':'productlist__link'}):
#             try:
#                 href = el.get('href')
#                 card_id = href.split('-')[-1].split('.html')[0][1:]
#                 links.append((href, card_id))
#             except:
#                 print(el)
#                 raise
#     print(len(links))
#     return links
# 
# def get_soup(url, params = {}):
#     tries = 10
#     sleep(10)
#     while True:
#         try:
#             resp = requests.get(url=url, params=params)
#             return BeautifulSoup(resp.text)
#         except:
#             print('connection problem, trying again...')
#             tries -= 1
#             if tries < 0:
#                 raise
# 
# def parse_on_gpu_data(link, gpu_id):
#     url = 'http://geizhals.eu/{0}'.format(link)
#     soup = get_soup(url)
#     desc = soup.find('div', {'id': 'gh_proddesc'})
#     # print(desc)
#     # print(desc.text)
#     data = desc.text.split(' &#149; ')
#     # print(data)
#     processor = data[0].split(': ')[1].split(' - ')[-1].split(' "')[0]
#     try:
#         memory = data[2].split(': ')[1].split(',')[0].split(' ')
#     except:
#         print('unviable data', data)
#         memory = None
#     dict_data = {
#         'prodId': gpu_id,
#         'name': soup.find('h1', {'class': 'arthdr'}).find('span', {'itemprop': 'name'}).string.split(', ')[0],
#         'processor': processor,
#         'memorycapacity': memory[0],
#         'memorytype': memory[1]
#     }
#     print(dict_data)
#     return dict_data
# 
# def parse_on_gpu_price(gpu_id):
#     url = 'http://geizhals.eu/'
#     hist_params = {
#         'phist': gpu_id,
#         'age': 9999
#     }
#     soup = get_soup(url, hist_params)
#     raw_data = []
#     parsed_data = []
#     scripts = soup.findAll('script')
#     for script in scripts:
#         if '_gh.plot' in script.text:
#             data = script.prettify().split('\n')
#             for line in data:
#                 if '_gh.plot' in line:
#                     idx = line.index('[')
#                     end = line.find('], {')
#                     price_hist = line[idx:end+1]
#                     raw_data = loads(price_hist)
#                     break
#     for point in raw_data:
#         price = point[1]
#         if price is None:
#             continue
#         price = float(price)
#         utc_time = datetime.utcfromtimestamp(point[0]/1000)
#         temp_dict = {
#             'prodId': gpu_id,
#             'date': utc_time.strftime("%Y-%m-%d"),
#             'price': price,
#             'region': 'de',
#             'merchant': 'geizhals_unknown'
#         }
#         parsed_data.append(temp_dict)
#     return parsed_data
# 
# def parse_on_cpu_data(link, gpu_id):
#     url = 'http://geizhals.eu/{0}'.format(link)
#     soup = get_soup(url)
#     desc = soup.find('div', {'id': 'gh_proddesc'})
#     # print(desc)
#     # print(desc.text)
#     data = desc.text.split(' &#149; ')
#     # print(data)
#     name = soup.find('h1', {'class': 'arthdr'}).find('span', {'itemprop': 'name'}).string.split(', ')[0]
#     # print(name)
#     proc_type = name.split(',')[0]
#     proc_cores = data[2].split(': ')[1].strip()
#     proc_socket = data[18].split(': ')[1].strip()
#     # processor = data[0].split(': ')[1].split(' - ')[-1].split(' "')[0]
#     # try:
#     #     memory = data[2].split(': ')[1].split(',')[0].split(' ')
#     # except:
#     #     print('unviable data', data)
#     #     memory = None
#     dict_data = {
#         'prodId': gpu_id,
#         'name': name,
#         'type': proc_type,
#         'cores': proc_cores,
#         'socket': proc_socket
#     }
#     print(dict_data)
#     return dict_data
# 
# def parse_on_cpu_price(gpu_id):
#     url = 'http://geizhals.eu/'
#     hist_params = {
#         'phist': gpu_id,
#         'age': 9999
#     }
#     soup = get_soup(url, hist_params)
#     raw_data = []
#     parsed_data = []
#     scripts = soup.findAll('script')
#     for script in scripts:
#         if '_gh.plot' in script.text:
#             data = script.prettify().split('\n')
#             for line in data:
#                 if '_gh.plot' in line:
#                     idx = line.index('[')
#                     end = line.find('], {')
#                     price_hist = line[idx:end+1]
#                     raw_data = loads(price_hist)
#                     break
#     for point in raw_data:
#         price = point[1]
#         if price is None:
#             continue
#         price = float(price)
#         utc_time = datetime.utcfromtimestamp(point[0]/1000)
#         temp_dict = {
#             'prodId': gpu_id,
#             'date': utc_time.strftime("%Y-%m-%d"),
#             'price': price,
#             'region': 'de',
#             'merchant': 'geizhals_unknown'
#         }
#         parsed_data.append(temp_dict)
#     return parsed_data
# 
# def parse_on_ram_data(link, gpu_id):
#     url = 'http://geizhals.eu/{0}'.format(link)
#     soup = get_soup(url)
#     desc = soup.find('div', {'id': 'gh_proddesc'})
#     # print(desc)
#     # print(desc.text)
#     data = desc.text.split(' &#149; ')
#     # print(data)
#     names = soup.find('h1', {'class': 'arthdr'}).find('span', {'itemprop': 'name'}).string.split(', ')
#     name = names[0]
#     print(names)
#     name_1 = names[1].split('-')
#     capacity = name.split(' ')[-1]
#     speed = name_1[1]
#     type_ddr = name_1[0]
#     # processor = data[0].split(': ')[1].split(' - ')[-1].split(' "')[0]
#     # try:
#     #     memory = data[2].split(': ')[1].split(',')[0].split(' ')
#     # except:
#     #     print('unviable data', data)
#     #     memory = None
#     dict_data = {
#         'prodId': gpu_id,
#         'name': name,
#         'type': type_ddr,
#         'speed': speed,
#         'capacity': capacity
#     }
#     print(dict_data)
#     return dict_data
# 
# def parse_on_ram_price(gpu_id):
#     url = 'http://geizhals.eu/'
#     hist_params = {
#         'phist': gpu_id,
#         'age': 9999
#     }
#     soup = get_soup(url, hist_params)
#     raw_data = []
#     parsed_data = []
#     scripts = soup.findAll('script')
#     for script in scripts:
#         if '_gh.plot' in script.text:
#             data = script.prettify().split('\n')
#             for line in data:
#                 if '_gh.plot' in line:
#                     idx = line.index('[')
#                     end = line.find('], {')
#                     price_hist = line[idx:end+1]
#                     raw_data = loads(price_hist)
#                     break
#     for point in raw_data:
#         price = point[1]
#         if price is None:
#             continue
#         price = float(price)
#         utc_time = datetime.utcfromtimestamp(point[0]/1000)
#         temp_dict = {
#             'prodId': gpu_id,
#             'date': utc_time.strftime("%Y-%m-%d"),
#             'price': price,
#             'region': 'de',
#             'merchant': 'geizhals_unknown'
#         }
#         parsed_data.append(temp_dict)
#     return parsed_data
# 
# def save_data(data, prices, component = 'GPU'):
#     if len(data) > 0:
#         save_csv_file('Data\\geizhals\\{0}_Types\\{1}.csv'.format(component, datetime.now().strftime("%Y%m%d_%H%M%S_%f")), data)
#     if len(prices) > 0:
#         save_csv_file('Data\\geizhals\\{0}_Prices\\{1}.csv'.format(component, datetime.now().strftime("%Y%m%d_%H%M%S_%f")), prices)
# 
# def load_data_from_site(links, comp_data_fun, comp_price_fun, component = 'GPU', skip=[]):
#     gpu_data = []
#     price_data = []
#     for idx, link in enumerate(links):
#         if idx+1 in skip:
#             print(idx+1, 'skipping')
#             continue 
#         print(idx+1, link[1])
#         try:
#             gpu_data.append(comp_data_fun(*link))
#             price_data.extend(comp_price_fun(link[1]))
#             if len(gpu_data) == 10:
#                 print(idx+1, 'saving data')
#                 save_data(gpu_data, price_data, component)
#                 price_data = []
#                 gpu_data = []
#         except:
#             if len(gpu_data) > 0 or len(price_data) > 0:
#                 save_data(gpu_data, price_data, component)
#                 print(idx+1, 'saving data because error occured')
#                 price_data = []
#                 gpu_data = []
#             else:
#                 print(idx+1, 'error occured')
#     save_data(gpu_data, price_data, component)
# 
# idx = []
# links = load_links_from_html('Data\\geizhals\\list_proc_1.html')
# load_data_from_site(links, parse_on_ram_data, parse_on_ram_price, 'CPU', idx)
# links = load_links_from_html('Data\\geizhals\\list_gpu_3.html')
# load_data_from_site(links, parse_on_ram_data, parse_on_ram_price, 'GPU', idx)
# links = load_links_from_html('Data\\geizhals\\list_ram_1.html')
# load_data_from_site(links, parse_on_ram_data, parse_on_ram_price, 'RAM', idx)
# ```

# ### PriceSpy
# -----
# ```python
# ### PriceSpy
# # -*- coding: utf8 -*-
# from csv import DictReader, DictWriter
# from datetime import datetime
# 
# import requests
# from BeautifulSoup import BeautifulSoup
# 
# def save_csv_file(file_name, data):
#     with open(file_name, 'w+b') as csvfile:
#         fieldnames = data[0].keys()
#         writer = DictWriter(csvfile, fieldnames=fieldnames)
#         writer.writeheader()
#         writer.writerows(data)
# 
# def load_ids_from_csv(file_name):
#     ids = []
#     with open(file_name, 'r') as csvfile:
#         reader = DictReader(csvfile)
#         ids = [row['prodId'] for row in reader]
#     return ids
# 
# def repair_pound_price(file_name):
#     with open(file_name, 'r') as csvfile:
#         reader = DictReader(csvfile)
#         data = [row for row in reader]
#         for d in data:
#             d['region'] = 'uk'
#             d['merchant'] = 'pricespy_unknown'
#     save_csv_file(file_name.replace('.csv', '_uk.csv'),data)
# 
# def load_gpu_ids_from_html(file_name):
#     parsed_data = []
#     ids = []
#     with open(file_name, 'r') as htmlfile:
#         soup = BeautifulSoup(htmlfile)
#         body = soup.find('tbody')
#         for el in body.findAll('tr'):
#             try:
#                 id_tag = el.get('id')
#                 if id_tag is None or not id_tag.startswith('erow_prod'):
#                     continue
#                 prodId = id_tag.split('-')[-1]
#                 prodName = el.findAll('a')[0].text
#                 tds = el.findAll('td')
#                 tds = tds[-4:-1]
#                 temp_dict = {
#                     'prodId': prodId,
#                     'name': prodName,
#                     'processor': tds[0].text.replace('&nbsp;', ' '),
#                     'memorycapacity': tds[1].text.replace('&nbsp;', ' '),
#                     'memorytype': tds[2].text.replace('&nbsp;', ' ')
#                 }
#                 ids.append(prodId)
#                 parsed_data.append(temp_dict)
#                 print(prodId, prodName)
#             except:
#                 print(el)
#                 raise
#             print('================================================================================')
#     print(len(parsed_data))
#     save_csv_file('Lab\Projekt\Data\pricespy\GPU_Types\{0}.csv'.format(datetime.now().strftime("%Y%m%d_%H%M%S_%f")), parsed_data)
#     return ids
# 
# def load_cpu_ids_from_html(file_name):
#     parsed_data = []
#     ids = []
#     with open(file_name, 'r') as htmlfile:
#         soup = BeautifulSoup(htmlfile)
#         body = soup.find('tbody')
#         for el in body.findAll('tr'):
#             try:
#                 id_tag = el.get('id')
#                 if id_tag is None or not id_tag.startswith('erow_prod'):
#                     continue
#                 prodId = id_tag.split('-')[-1]
#                 prodName = el.findAll('a')[0].text
#                 tds = el.findAll('td')
#                 tds = tds[-4:-1]
#                 temp_dict = {
#                     'prodId': prodId,
#                     'name': prodName,
#                     'type': tds[0].text.replace('&nbsp;', ' '),
#                     'socket': tds[1].text.replace('&nbsp;', ' '),
#                     'cores': tds[2].text.replace('&nbsp;', ' ').strip()
#                 }
#                 ids.append(prodId)
#                 parsed_data.append(temp_dict)
#                 print(prodId, prodName)
#             except:
#                 print(el)
#                 raise
#             print('================================================================================')
#     print(len(parsed_data))
#     save_csv_file('Lab\Projekt\Data\pricespy\CPU_Types\{0}.csv'.format(datetime.now().strftime("%Y%m%d_%H%M%S_%f")), parsed_data)
#     return ids
# 
# def load_ram_ids_from_html(file_name, ram_type):
#     parsed_data = []
#     ids = []
#     with open(file_name, 'r') as htmlfile:
#         soup = BeautifulSoup(htmlfile)
#         body = soup.find('tbody')
#         for el in body.findAll('tr'):
#             try:
#                 id_tag = el.get('id')
#                 if id_tag is None or not id_tag.startswith('erow_prod'):
#                     continue
#                 prodId = id_tag.split('-')[-1]
#                 prodName = el.findAll('a')[0].text
#                 tds = el.findAll('td')
#                 tds = tds[-4:-1]
#                 temp_dict = {
#                     'prodId': prodId,
#                     'name': prodName,
#                     'type': ram_type,
#                     'capacity': tds[0].text.replace('&nbsp;', ' '),
#                     'speed': tds[1].text.replace('&nbsp;', ' ')
#                 }
#                 ids.append(prodId)
#                 parsed_data.append(temp_dict)
#                 print(prodId, prodName)
#             except:
#                 print(el)
#                 raise
#             print('================================================================================')
#     print(len(parsed_data))
#     save_csv_file('Lab\Projekt\Data\pricespy\RAM_Types\{0}.csv'.format(datetime.now().strftime("%Y%m%d_%H%M%S_%f")), parsed_data)
#     return ids
# 
# def scrap_data_history_from_graph_price(ids, comp_name = 'GPU'):
#     scrap_data_history_from_graph(ids, 'price_history', 'price', 'Lab\Projekt\Data\pricespy\{0}_Prices\{1}.csv'.format(comp_name, datetime.now().strftime("%Y%m%d_%H%M%S_%f")))
# 
# # def scrap_data_history_from_graph_popularity(ids):
# #     scrap_data_history_from_graph(ids, 'popularity_history', 'popularity', 'Lab\Projekt\Data\GPU_Popularity\{0}.csv'.format(datetime.now().strftime("%Y%m%d_%H%M%S_%f")))
# 
# def scrap_data_history_from_graph(ids, method, value_tag, file_name):
#     url = 'https://pricespy.co.uk/ajax/server.php'
#     params = {
#         'class':'Graph_Product',
#         'method':method,
#         'skip_login':1,
#         'product_id':None
#     }
#     parsed_data = []
#     for idx, prodId in enumerate(ids):
#         print(idx, prodId)
#         params['product_id'] = prodId
#         resp = requests.get(url=url, params=params)
#         data = resp.json()
#         for item in data['items'][0]:
#             utc_time = datetime.utcfromtimestamp(item['time'])
#             temp_dict = {
#                 'prodId': prodId,
#                 'date': utc_time.strftime("%Y-%m-%d"),
#                 value_tag: item['value'],
#                 'region': 'uk',
#                 'merchant': 'pricespy_unknown'
#             }
#             parsed_data.append(temp_dict)
#     save_csv_file(file_name, parsed_data)
# 
# ids = load_gpu_ids_from_html('Data\\pricespy\\lista_gpu_1.html')
# print(len(ids))
# scrap_data_history_from_graph_price(ids, 'GPU')
# 
# ids = load_cpu_ids_from_html('Data\\pricespy\\lista_proc_1.html')
# print(len(ids))
# scrap_data_history_from_graph_price(ids, 'CPU')
# 
# ids = load_ram_ids_from_html('Data\\pricespy\\lista_ddr4_1.html', 'DDR4')
# print(len(ids))
# scrap_data_history_from_graph_price(ids, 'RAM')
# 
# ids = load_ram_ids_from_html('Data\\pricespy\\lista_ddr3_1.html', 'DDR3')
# print(len(ids))
# scrap_data_history_from_graph_price(ids, 'RAM')
# 
# ids = load_ram_ids_from_html('Data\\pricespy\\lista_ddr2_1.html', 'DDR2')
# print(len(ids))
# scrap_data_history_from_graph_price(ids, 'RAM')
# ```

# #### CSV Concatenation
# -----
# ```python
# ### Concatenate CSV files into one
# # -*- coding: utf8 -*-
# from unicodecsv import DictReader, DictWriter
# from datetime import datetime
# 
# def save_csv_file(file_name, data):
#     if len(data) > 0:
#         with open(file_name, 'w+b') as csvfile:
#             fieldnames = data[0].keys()
#             writer = DictWriter(csvfile, fieldnames=fieldnames)
#             writer.writeheader()
#             writer.writerows(data)
# 
# def concat_csv_prices_files(directory, folder, component):
#     files = os.listdir(directory)
#     data = []
#     counter = 0
#     for file_name in files:
#         if folder == 'pricespy' and component == 'GPU' and not 'uk' in file_name:
#             continue
#         temp_counter = 0
#         with open('{0}\{1}'.format(directory,file_name), 'r') as csvfile:
#             reader = DictReader(csvfile)
#             for row in reader:
#                 data.append(row)
#                 temp_counter += 1
#         counter += temp_counter
#         print(file_name, temp_counter, counter)
#     print("saving data")
#     save_csv_file('Data\\{0}\\{1}_Prices_merged\\{2}_merged_{3}.csv'.format(folder, component, datetime.now().strftime("%Y%m%d_%H%M%S_%f"), counter), data)
# 
# folders = ['pricespy', 'pcpartpicker', 'geizhals']
# components = ['GPU', 'CPU', 'RAM']
# for folder in folders:
#     for component in components:
#         directory_type = 'Data\\{0}\\{1}_Types'.format(folder, component)
#         directory_price = 'Data\\{0}\\{1}_Prices'.format(folder, component)
#         concat_csv_prices_files(directory_price, folder, component)
# ```
