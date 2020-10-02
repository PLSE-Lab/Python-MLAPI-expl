import sqlite3

data = sqlite3.connect('../input/database.sqlite')

print('Found one: ', type(data))

found_juggling = "Select lower(body)    \
    From May2015                        \
    Where LENGTH(body) < 40             \
    and LENGTH(body) > 20               \
    and lower(body) LIKE 'juggling%'    \
    Limit 100";

print(type(found_juggling))
print(len(found_juggling))
print(found_juggling)