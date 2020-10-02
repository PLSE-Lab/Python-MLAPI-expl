# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

import io
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn import metrics
import numpy as np
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
from time import time

def convert_pdf_to_txt(path):
    rsrcmgr = PDFResourceManager()
    retstr = io.StringIO()
    codec = 'utf-8'
    laparams = LAParams()
    device = TextConverter(rsrcmgr, retstr, codec=codec, laparams=laparams)
    fp = open(path, 'rb')
    interpreter = PDFPageInterpreter(rsrcmgr, device)
    password = ""
    maxpages = 0
    caching = True
    pagenos = set()

    for page in PDFPage.get_pages(fp, pagenos, maxpages=maxpages,
                                  password=password,
                                  caching=caching,
                                  check_extractable=True):
        interpreter.process_page(page)

    text = retstr.getvalue()

    fp.close()
    device.close()
    retstr.close()
    return text


#ds1 = convert_pdf_to_txt("c:\\Users\\bjerre\\desktop\\ds1.pdf")
#remiss1 = convert_pdf_to_txt("c:\\Users\\bjerre\\desktop\\remiss1.pdf")
prop1 = convert_pdf_to_txt("../input/prop1.pdf")
dir2 = convert_pdf_to_txt("../input/dir2.pdf")
sou2 = convert_pdf_to_txt("../input/sou2.pdf")
remiss2 = convert_pdf_to_txt("../input/remiss2.pdf")
prop2 = convert_pdf_to_txt("../input/prop2.pdf")
lagradsremiss3 = convert_pdf_to_txt("../input/lagradsremiss3.pdf")
prop3 = convert_pdf_to_txt("../input/prop3.pdf")
sou4 = convert_pdf_to_txt("../input/sou4.pdf")
dir4 = convert_pdf_to_txt("../input/dir4.pdf")
sou42 = convert_pdf_to_txt("../input/sou4-2.pdf")

vectorizer = TfidfVectorizer(max_df=1,
                                 min_df=1,
                                 use_idf=True)
X = np.array([prop1, dir2, sou2, remiss2, prop2, lagradsremiss3, prop3, sou4, dir4, sou42])
y = np.array([0,1,1,1,1,2,2,3,3,3])
print(X.shape)
X = vectorizer.fit_transform(X)
km = KMeans(n_clusters=4, init='k-means++', max_iter=100, n_init=1, verbose=True)
km.fit(X)
print(km.labels_)
print(km.predict(vectorizer.transform([prop2])))

svd = TruncatedSVD(n_components=4, n_iter=7, random_state=42)
svd.fit(X)
X_reduced = svd.fit_transform(X)
print(svd.explained_variance_ratio_)
print(svd.explained_variance_ratio_.sum())
print(svd.singular_values_)

ax = plt.subplot(222)
ax.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y, s=50, edgecolor='k')
ax.set_title("Truncated SVD reduction (2d) of transformed data (%dd)" %
             X.shape[1])
ax.set_xticks(())
ax.set_yticks(())

plt.tight_layout()
plt.show()
