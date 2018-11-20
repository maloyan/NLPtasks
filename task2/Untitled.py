
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import os
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer


# In[2]:


#0 - русский 1 - белорусский 2 - украинский 3 - болгарский 4 - македонский 5 - сербский
languages_dict = {'ru' : 0, 'be' : 1, 'uk' : 2, 'bg' : 3, 'mk' : 4, 'sr' : 5}


# In[4]:


ru = pd.read_csv('./data/ru.csv')
be = pd.read_csv('./data/be.csv')
uk = pd.read_csv('./data/uk.csv')
bg = pd.read_csv('./data/bg.csv')
mk = pd.read_csv('./data/mk.csv')
sr = pd.read_csv('./data/sr.csv')


# In[5]:


df = ru.head(12000)
df = df.append(be.head(12000))
df = df.append(uk.head(12000))
df = df.append(bg.head(12000))
df = df.append(mk.head(12000))
df = df.append(sr.head(12000))


# In[6]:


df_test = pd.read_csv('./data/task2-dev.csv')


# In[7]:


Y_train = df['language']
Y_test  = df_test['language']


# In[8]:


n = df.shape[0]
vec = TfidfVectorizer(ngram_range=(1,4), analyzer='char')
X_train = vec.fit_transform(df['text'])
X_test  = vec.transform(df_test['text']) 


# <h1>SVM</h1>

# In[9]:


from sklearn import svm


# In[10]:


clf1 = svm.LinearSVC()
clf1.fit(X_train, Y_train)
y_pred = clf1.predict(X_test)
print(accuracy_score(Y_test, y_pred))


# In[11]:


path = os.path.join("./data/test_no_labels.txt")
text = []
with open(path, 'r', encoding = "utf-8") as f:
    for line in f:
        text.append(line[:-1])


# In[21]:


X_submit = pd.DataFrame(columns=['language', 'text'])


# In[24]:


X_submit['text'] = text


# In[14]:


X_sumbit_trans = vec.transform(X_submit['text']) 


# In[16]:


Y_submit = clf1.predict(X_sumbit_trans)


# In[22]:


X_submit['language'] = Y_submit


# In[26]:


X_submit.to_csv('./submit.csv', encoding='utf-8', index=False)

