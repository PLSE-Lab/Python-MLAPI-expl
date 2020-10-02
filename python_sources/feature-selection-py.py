import re
import numpy as np
from scipy import sparse
from sklearn import svm, model_selection, metrics, preprocessing

splitter = re.compile(r"\W")
# We detect the encodings below using chardet
encodings = (
    "ascii",
    "windows-1252",
    "ISO-8859-2",
    "KOI8-R",
)

def decode(text):
    for encoding in encodings:
        try:
            return text.decode(encoding)
        except UnicodeDecodeError:
            pass
    return None

sp_rows = []
sp_cols = []
sp_values = []
y = []
token2id = {}
label_dict = {"spam": 1, "ham": 0}
i = 0
with open("../input/spam.csv", "rb") as fp:
    fp.readline()
    for row in fp:
        row = row.strip()
        row = decode(row)
        if not row:
            continue
        columns = row.split(",")
        label = label_dict[columns[0]]
        text = ",".join(columns[1:])
        for token in splitter.split(text):
            if token:
                j = token2id.setdefault(token.lower(), len(token2id))
                sp_rows.append(i)
                sp_cols.append(j)
                sp_values.append(1.0)
        y.append(label)
        i += 1

X = sparse.csr_matrix((sp_values, (sp_rows, sp_cols)), shape=(i, len(token2id)))
X = preprocessing.normalize(X, axis=1)
y = np.array(y)

print("n_samples:", X.shape[0])
print("n_features:", X.shape[1])

random_state = 1
n_splits = 5
Cs = np.logspace(0, 3, 10)

best_C = None
best_score = -np.inf
skf = model_selection.StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
for C in Cs:
    scores = np.zeros(n_splits)
    for i, (itr, ite) in enumerate(skf.split(X, y)):
        svc = svm.LinearSVC(penalty="l1", dual=False, C=C, random_state=random_state)
        svc.fit(X[itr], y[itr])
        ypred = svc.predict(X[ite])
        scores[i] = metrics.accuracy_score(y[ite], ypred)
    this_score = scores.mean()
    if this_score > best_score:
        best_score = this_score
        best_C = C

print("%d-fold accuracy: %.3f, best C: %f" % (n_splits, this_score, best_C))
svc = svm.LinearSVC(penalty="l1", dual=False, C=best_C, random_state=random_state)
svc.fit(X, y)

id2token = {value: key for key, value in token2id.items()}

coef = svc.coef_[0]

topn = 20
print()
print("top-%d predicted spam-related tokens:" % topn)
for i in coef.argsort()[::-1][:topn]:
    print(id2token[i], coef[i])

# output
# n_samples: 5572
# n_features: 8801
# 5-fold accuracy: 0.984, best C: 215.443469
#
# top-20 predicted spam-related tokens:
# 01223585236 2.94496597862
# 07090201529 2.394468371
# 146tf150p 2.03107339744
# 84484 1.75027669096
# callså 1.72431174878
# wining 1.55797659998
# xafter 1.5555316889
# accordingly 1.53470951797
# ringtone 1.39275147302
# tbs 1.35721192949
# fgkslpopw 1.34970020281
# 150p 1.31190339663
# dartboard 1.25773898409
# pause 1.2459664777
# trial 1.22136859538
# thousands 1.20435502674
# simpsons 1.19894894059
# gmw 1.17183463408
# immediately 1.15613085601
# divorce 1.14126732654
