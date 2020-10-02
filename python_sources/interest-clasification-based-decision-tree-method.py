#!/usr/bin/env python
# coding: utf-8

# In[ ]:


print("Machine Learning for Interest Classification System based Decision Tree Method")


# In[ ]:


from sklearn import tree


# In[ ]:


#Data training untuk Machine Learning

a = [
        [4,2,1,2,2],
        [1,1,4,1,1],
        [3,1,2,2,3],
        [4,1,4,1,5],
        [4,3,3,1,6],
        [4,1,3,1,1],
        [4,3,3,1,7],
        [4,1,1,1,3],
        [4,1,2,1,6],
        [3,1,1,1,6],
        [2,1,1,1,7],
        [3,1,1,1,3],
        [2,1,1,1,6],
        [3,1,1,2,4],
        [3,2,1,2,7],
        [4,1,1,2,2],
        [2,2,1,2,3],
        [4,2,1,1,7],
        [3,3,1,1,3],
        [1,1,2,1,3]
    ]


b = [
     'politik',
     'sains dan budaya',
     'sains',
     'sains',
     'Budaya',
     'sains',
     'Budaya',
     'politik dan budaya',
     'sains',
     'budaya',
     'politik',
     'sains dan budaya',
     'sains',
     'politik',
     'budaya',
     'politik',
     'sains',
     'politik',
     'budaya',
     'sains' 
    ]


# In[ ]:


clf = tree.DecisionTreeClassifier()

clf = clf.fit(a,b)


# ##Prediksi minat konsumen dengan parameter :
# 
# 1. Semester 1 kuliah (maba)
# 2. Jurusan Fisika
# 3. Aktif di dunia riset ilmiah
# 4. berasal dari daerah kabupaten Malang
# 5. senang membaca buku sejarah

# In[ ]:


prediksi = clf.predict([[2,1,4,1,2]])
prediksi


# Didapatkan kesimpulan bahwa konsumen tersebut lebih berminat dengan "sains dan budaya" daripada politik
