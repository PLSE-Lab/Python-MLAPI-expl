import wikipedia

query = "india"

res = wikipedia.summary(query, sentences=5)

print(res)