
# coding: utf-8

# # Инициализация

# In[1]:


import os
import numpy as np
import pymorphy2
from tqdm import tqdm_notebook


# In[2]:


token_filenames = [f for f in os.listdir('./devset/') if '.tokens' in f]


# # Функции для работы с файлами

# In[3]:


class Token:
    def __init__(self, position, length, text):
        self._position = position
        self._length = length
        self._text = text
        self._pos = None
        self._tag = None


# In[4]:


class Span:
    def __init__(self, token_id):
        self._token_id = token_id


# In[5]:


def load_tokens(token_filename, dev=True):
    _set = './devset/' if dev else './testset/'
    tokens = dict()
    with open(_set + token_filename, encoding='utf8') as f:
        for line in f:
            split = line.split()
            if len(split) > 0:
                t = Token(split[1], split[2], split[3])
                tokens[split[0]] = t
    return tokens


# In[6]:


def load_spans(token_filename, dev=True):
    _set = './devset/' if dev else './testset/'
    spans = dict()
    with open(_set + token_filename.split('.')[0] + '.spans', encoding='utf8') as f:
        for line in f:
            split = line.split()
            s = Span(split[4])
            spans[split[0]] = s
    return spans


# In[7]:


def transform_base_tag(base_tag):
    if base_tag == 'Person':
        return 'PER'
    if base_tag == 'Location' or base_tag == 'LocOrg':
        return 'LOC'
    if base_tag == 'Org':
        return 'ORG'
    else:
        return 'MISC'


# In[8]:


def load_objects(token_filename, tokens, spans, dev=True):
    _set = './devset/' if dev else './testset/'
    with open(_set + token_filename.split('.')[0] + '.objects', encoding='utf8') as f:
        for line in f:
            line = line.split(' # ')[0]
            split = line.split()
            base_tag = transform_base_tag(split[1])
            span_ids = split[2:]
            if len(span_ids) == 1:
                tokens[spans[span_ids[0]]._token_id]._tag = 'U-' + base_tag
            else:
                for i, span_id in enumerate(span_ids):
                    if i == 0:
                        tokens[spans[span_ids[i]]._token_id]._tag = 'B-' + base_tag
                    if i == len(span_ids) - 1:
                        tokens[spans[span_ids[i]]._token_id]._tag = 'L-' + base_tag
                    else:
                        tokens[spans[span_ids[i]]._token_id]._tag = 'I-' + base_tag
    return tokens


# In[9]:


morph = pymorphy2.MorphAnalyzer()


# In[10]:


def fill_pos(tokens):
    for id, token in tokens.items():
        pos = morph.parse(token._text)[0].tag.POS
        if pos is None:
            pos = 'None'
        token._pos = pos
        if token._tag is None:
            token._tag = 'O'
    return tokens


# In[11]:


def word2features(sent, i):
    word = sent[i]._text
    postag = sent[i]._pos

    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        'postag': postag,
    }
    if i > 0:
        word1 = sent[i-1]._text
        postag1 = sent[i-1]._pos
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
            '-1:postag': postag1,
        })
    else:
        features['BOS'] = True

    if i < len(sent)-1:
        word1 = sent[i+1]._text
        postag1 = sent[i+1]._pos
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
            '+1:postag': postag1,
        })
    else:
        features['EOS'] = True

    return features


def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

def sent2labels(sent):
    return [token._tag for token in sent]

def sent2tokens(sent):
    return [token for token in sent]


# In[12]:


def split_tokens_by_sents(tokens):
    sents = []
    sent = []
    for id, token in tokens.items():
        if token._text != '.':
            sent.append(token)
        else:
            sents.append(sent)
            sent = []
    return sents


# # Готовим тренировочную выборку

# In[13]:


def generate_sents(token_filename):
    tokens = load_tokens(token_filename)
    spans = load_spans(token_filename)
    tokens = load_objects(token_filename, tokens, spans)
    tokens = fill_pos(tokens)
    sents = split_tokens_by_sents(tokens)
    return sents


# In[14]:


sents = []
for token_filename in tqdm_notebook(token_filenames):
    sents += generate_sents(token_filename)


# In[15]:


len(sents)


# In[16]:


from sklearn.model_selection import train_test_split
import numpy as np

train_ids, test_ids = train_test_split(np.arange(len(sents)))


# In[17]:


train_sents = np.array(sents)[train_ids]
test_sents = np.array(sents)[test_ids]


# In[18]:


X_train = [sent2features(s) for s in train_sents]
y_train = [sent2labels(s) for s in train_sents]

X_test = [sent2features(s) for s in test_sents]
y_test = [sent2labels(s) for s in test_sents]


# In[27]:


X_train


# # Тренируем модель

# In[20]:


from sklearn_crfsuite import CRF

crf = CRF(algorithm='lbfgs', c1=0.1, c2=0.1, max_iterations=100, all_possible_transitions=True)
crf.fit(X_train, y_train)


# In[21]:


from sklearn_crfsuite.metrics import flat_classification_report

print(flat_classification_report(y_test, crf.predict(X_test)))


# # Применяем модель

# In[22]:


def get_entities(tokens):
    rows = []

    buffer = []
    for id, token in tokens.items():
        tag = token._tag
        if tag.startswith('U'):
            rows.append('%s %d %d\n' % (tag.split('-')[1], int(token._position), int(token._length)))
        elif tag.startswith('B') or tag.startswith('I'):
            buffer.append(token)
        elif tag.startswith('L'):
            buffer.append(token)
            start = int(buffer[0]._position)
            length = int(buffer[-1]._position) + int(buffer[-1]._length) - int(start)
            rows.append('%s %d %d\n' % (tag.split('-')[1], start, length))
            buffer = []
    return rows


# In[23]:


test_token_filenames = [filename for filename in os.listdir('./testset') if '.tokens' in filename]


# In[24]:


for token_filename in tqdm_notebook(test_token_filenames):
    tokens = load_tokens(token_filename, dev=False)
    tokens = fill_pos(tokens)
    sents = split_tokens_by_sents(tokens)
    X = [sent2features(s) for s in sents]
    y_pred = crf.predict(X)
    for i in range(len(y_pred)):
        for j in range(len(y_pred[i])):
            sents[i][j]._tag = y_pred[i][j]
    rows = get_entities(tokens)
    with open('./results_crf/' + token_filename.split('.')[0] + '.task1', 'w') as f:
        f.writelines(rows)


# # Проверяем результаты

# In[25]:


#get_ipython().system('python scripts\\t1_eval.py -s .\\testset -t .\\results_crf -o .\\output')

