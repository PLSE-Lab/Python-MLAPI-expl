import pandas as pd

def union(a, b):
    return list(set(a) | set(b))

def get_freq(A):
	vals = {}
	for a in A:
		if a in vals:
			vals[a] += 1.0
		else:
			vals[a] = 1.0
	for k in vals:
		vals[k] /= len(A)
	return vals

# remove characters like "'" or "."
def wordify(t):
	t = t.replace("'", " ")
	t = t.replace(",", " ")
	t = t.replace(".", " ")
	t = t.replace("!", " ")
	t = t.replace("?", " ")
	t = t.replace("&", " ")
	t = t.replace("|", " ")
	t = t.replace("/", " ")
	t = t.replace(";", " ")
	t = t.replace(":", " ")
	t = t.replace("\\", " ")
	t = t.replace("\\n", " ")
	return t

def try_str(t):
	try:
		return str(t)
	except:
		return ""

INPUT_CSV = "../input/gender-classifier-DFE-791531.csv"

csv = pd.read_csv(INPUT_CSV, encoding='latin1')
header = csv.head()
rows = csv.values.tolist()

# remove points we're not confident about
rows = list(filter(lambda x: x[6] == 1.0, rows))

# split into 'male' and 'female' sets:
males = list(filter(lambda x: x[5] == "male", rows))
females = list(filter(lambda x: x[5] == "female", rows))
brands = list(filter(lambda x: x[5] == "brand", rows))

print (str(len(females)) + " females")
print (str(len(males)) + " males")
print (str(len(brands)) + " brands")


# combine female descriptions into a single list of 'words'
female_descriptions = " ".join(list(map(lambda x: try_str(x[10]), females)))
female_descriptions = female_descriptions.lower()
female_descriptions = wordify(female_descriptions).split(" ")
female_descriptions = list(w for w in female_descriptions if len(w) > 1)

# same for males' descriptions
male_descriptions = " ".join(list(map(lambda x: try_str(x[10]), males)))
male_descriptions = male_descriptions.lower()
male_descriptions = wordify(male_descriptions).split(" ")
male_descriptions = list(w for w in male_descriptions if len(w) > 1)

# same for brands' descriptions
brand_descriptions = " ".join(list(map(lambda x: try_str(x[10]), brands)))
brand_descriptions = brand_descriptions.lower()
brand_descriptions = wordify(brand_descriptions).split(" ")
brand_descriptions = list(w for w in brand_descriptions if len(w) > 1)

A = get_freq(female_descriptions)
B = get_freq(male_descriptions)
C = get_freq(brand_descriptions)
keys = union(union(A.keys(), B.keys()), C.keys())

relative_f = []
relative_m = []
relative_b = []
for k in keys:
	if k not in A:
		A[k] = 0.0
	if k not in B:
		B[k] = 0.0
	if k not in C:
		C[k] = 0.0
	if A[k] + B[k] + C[k] > 0.002:
		relative_f.append((A[k] / (A[k] + B[k] + C[k]), k))
		relative_m.append((B[k] / (A[k] + B[k] + C[k]), k))
		relative_b.append((C[k] / (A[k] + B[k] + C[k]), k))
relative_f = list(sorted(relative_f))
relative_m = list(sorted(relative_m))
relative_b = list(sorted(relative_b))

print ("\nfemale:")
for w in relative_f[-10:]:
	print (w)

print ("male:")
for w in relative_m[-10:]:
	print (w)

print ("brand:")
for w in relative_b[-10:]:
	print (w)
