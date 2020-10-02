#!/usr/bin/env python3

import sys

print("Id,Category")
count = 1
for line in sys.stdin:
    parts = line.strip().split('\t')
    name = parts[0].lower()
    r = ''
    if name.find('facebook') != -1:
        r = '/wiki/Facebook'
    elif name.find('youtube') != -1:
        r = '/wiki/YouTube'
    elif name.find('google') != -1:
        r = '/wiki/Google'
    elif name.find('instagram') != -1:
        r = '/wiki/Instagram'
    else:
        r = '/wiki/'
    print(str(count) + ',"' + r + '"')
    count += 1

