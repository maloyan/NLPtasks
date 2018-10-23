
# coding: utf-8

# In[3]:


import os
import pymorphy2
import re
import string
import numpy as np
from collections import Counter
from sklearn import preprocessing


# In[4]:


DEV_PATH  = './devset/'
TEST_PATH = './testset/'


# In[5]:


filenames_dev  = [f.split('.')[0] for f in os.listdir(DEV_PATH) if '.tokens' in f]
filenames_test = [f.split('.')[0] for f in os.listdir(TEST_PATH) if '.tokens' in f]


# In[6]:


def load_tokens(path = DEV_PATH, filenames = filenames_dev):
    tokens = dict()
    for file in filenames:
        with open(path + file + '.tokens', 'r+', encoding='utf8') as f:
            for line in f:
                split = line.split()
                if split:
                    tokens[split[0]] = split[1:]
    return tokens


# In[7]:


def load_spans(path = DEV_PATH, filenames = filenames_dev):
    spans = dict()
    for file in filenames:
        with open(path + file + '.spans', 'r+', encoding='utf8') as f:
            for line in f:
                split = line.split()
                spans[split[0]] = split[1:]
    return spans


# In[8]:


def load_objects(path = DEV_PATH, filenames = filenames_dev):
    objects = dict()
    for file in filenames:
        with open(path + file + '.objects', 'r+', encoding='utf8') as f:
            for line in f:
                part = line.split(' # ')[0]
                split = part.split()
                if split[1] == 'Location' or split[1] == 'LocOrg':
                    objects[split[0]] = split[1:]
    return objects


# In[9]:


def parse(path = DEV_PATH, filenames = filenames_dev):
    tokens = load_tokens(path, filenames)
    spans  = load_spans(path, filenames)
    objects = load_objects(path, filenames)
    for key, value in objects.items():
        ne = value[0]
        span_ids = value[1:]
        for i, span_id in enumerate(span_ids):
            tokens[spans[span_ids[i]][3]].append(ne)
    
    token_list = []         
    for key, value in tokens.items():
        if len(value) == 3:
            value.append('O')
        token_list.append(value)
    return token_list


# In[10]:


POS  = 0
LEN  = 1
WORD = 2
NE   = 3
CTX_LEN = 2


# In[11]:


feature = parse()


# In[12]:


feature


# In[13]:


#Выделим только слова
data = []
for f in feature:
    data.append(f[WORD])
    
# Добавим пустые слова для контекстной информации
data = ["" for i in range(CTX_LEN)] + data
data = data + ["" for i in range(CTX_LEN)]


# In[14]:


# POS-тег слова #
morph = pymorphy2.MorphAnalyzer()
def get_pos(token):
    pos = morph.parse(token)[0].tag.POS
    if pos:
        return pos
    return None


# In[15]:


# Тип регистра слова #
def get_capital(token):
    pattern = re.compile("[{}]+$".format(re.escape(string.punctuation)))
    if pattern.match(token):
        return "none"
    if len(token) == 0:
        return "none"
    if token.islower():
        return "lower"
    elif token.isupper():
        return "upper"
    elif token[0].isupper() and len(token) == 1:
        return "proper"
    elif token[0].isupper() and token[1:].islower():
        return "proper"
    else:
        return "camel"


# In[16]:


# Признак того, является ли слово пунктуацией #
def get_is_punct(token):
    pattern = re.compile("[{}]+$".format(re.escape(string.punctuation)))
    if pattern.match(token):
        return "yes"
    else:
        return "no"


# In[17]:


# Признак того, является ли слово числом #
def get_is_number(token):
    try:
        complex(token)
    except ValueError:
        return "no"
    return "yes"


# In[18]:


# Возвращает начальную форму слова #
def get_initial(token):
    init = morph.parse(token)[0].normal_form
    if init:
        return init
    else:
        return None


# In[19]:

features_list = []
for k in range(len(data) - 2 * CTX_LEN):
    arr = []
    i = k + CTX_LEN
    print(k)
    pos_arr = [get_pos(data[i])]
    capital_arr = [get_capital(data[i])]
    initial_arr = [get_initial(data[i])]

    for j in range(1, CTX_LEN + 1):
        pos_arr.append(get_pos(data[i - j]))
        pos_arr.append(get_pos(data[i + j]))

        capital_arr.append(get_capital(data[i - j]))
        capital_arr.append(get_capital(data[i + j]))

        initial_arr.append(get_initial(data[i - j]))
        initial_arr.append(get_initial(data[i + j]))

    arr += pos_arr
    arr += capital_arr
    arr += initial_arr

    features_list.append(arr)


# In[20]:

print("Hello")
features_list = np.array([np.array(line) for line in features_list])


# In[21]:


# Выкинем из этого массива классы, встретившиеся менее NUMBER_OF_OCCURENCES раз #
# Посчитаем частоту лейблов в столбце #
number_of_columns = features_list.shape[1]
counters = []
for u in range(number_of_columns):
    arr = features_list[:, u]
    counter = Counter(arr)
    counters.append(counter)


# In[22]:


# Заменяет лейбл на "*", если он "редкий" #
NUMBER_OF_OCC = 5
def get_feature(f, feature):
    if feature in counters[f].keys() and counters[f][feature] > NUMBER_OF_OCC:
        return feature
    else:
        return "*"


# In[23]:


# Избавимся от редких лейблов (частота < NUMBER_OF_OCC) #
for y in range(len(features_list)):
    for x in range(number_of_columns):
        features_list[y][x] = get_feature(x, features_list[y][x])


# In[24]:


# Переводит категории в числовое представление #
class ColumnApplier(object):
    def __init__(self, column_stages):
        self._column_stages = column_stages

    def fit(self, x, y):
        for i, k in self._column_stages.items():
            k.fit(x[:, i])
        return self

    def transform(self, x):
        x = x.copy()
        for i, k in self._column_stages.items():
            x[:, i] = k.transform(x[:, i])
        return x


# In[25]:


multi_encoder = ColumnApplier(dict([(i, preprocessing.LabelEncoder()) for i in range(len(features_list[0]))]))
features_list = multi_encoder.fit(features_list, None).transform(features_list)


# In[26]:


enc = preprocessing.OneHotEncoder(dtype=np.bool_, sparse=True)
enc.fit(features_list)
features_list = enc.transform(features_list)

