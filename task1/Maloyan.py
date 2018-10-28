import sys
sys.path.append('scripts/dialent/task1/')
sys.path.append('scripts/dialent/')
sys.path.append('scripts/')
import util
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import xgboost as xgb
import re, string


re_tok = re.compile(f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])')
CTX = 3
best_parameters = {
    'learning_rate': 0.08,
    'max_depth': 5
}

def tokenize(s): return re_tok.sub(r' \1 ', s).split()

def tkn_dict(u):
    d = dict()
    for i in range(len(u)):
        tokens = u[i].tokens
        for j in tokens:
            d[int(j.id)] = j.text
    return d

def ne_bin(x):
    if x == 'LOC':
        return 0
    elif x == 'LOCORG':
        return 1
    return 2

def get_df(u):
    df = pd.DataFrame()
    for i in range(len(u)):
        res = u[i].makeTokenSets()
        for j in range(len(res)):
            token = res[j].toInlineString()
            split = token.split()
            token_id = int(str(res[j]).split('#')[-1][:-2])
            df = df.append({'tokens': split[4:], 
                            'tokens_id': token_id,
                            'obj_id': int(split[1]),
                            'ne_bin': ne_bin(str(split[0]))}, ignore_index=True)
    return df

def get_from_dict_dev(id_d):
    d = tkn_dev_dict
    ans = str()
    for i in range(-CTX, CTX):
        if id_d + i in d:
            ans += " " + str(d[id_d + i])
    return ans

def get_from_dict_test(id_d):
    d = tkn_test_dict
    ans = str()
    for i in range(-CTX, CTX):
        if id_d + i in d:
            ans += " " + str(d[id_d + i])
    return ans

u_dev = util.loadAllStandard('./devset/')
u_test = util.loadAllStandard('./testset/')

tkn_dev_dict = tkn_dict(u_dev)
tkn_test_dict = tkn_dict(u_test)

df_dev = get_df(u_dev)
df_dev['ctx'] = df_dev['tokens_id'].apply(get_from_dict_dev)
df_dev = df_dev[df_dev['ne_bin'] < 2]

df_test = get_df(u_test)
df_test['ctx'] = df_test['tokens_id'].apply(get_from_dict_test)
df_test = df_test[df_test['ne_bin'] < 2]

n = df_dev.shape[0]
vec = TfidfVectorizer(ngram_range=(1, 2), tokenizer=tokenize,
                      min_df=3, max_df=0.9, strip_accents='unicode',
                      use_idf=1, smooth_idf=1, sublinear_tf=1)
X_dev = vec.fit_transform(df_dev['ctx'])
X_test = vec.transform(df_test['ctx'])

Y_dev = df_dev['ne_bin']
Y_test = df_test['ne_bin']

clf = xgb.XGBClassifier(**best_parameters)
clf.fit(X_dev, Y_dev)
predict = clf.predict(X_test)

u = util.loadAllStandard('./testset/')
for i in range(len(u)):
    res = u[i].makeTokenSets()
    row = []
    for j in range(len(res)):
        token = res[j].toInlineString()
        if token[0] == 'L':
            split = token.split()
            if predict[df_test['obj_id'] == int(split[1])][0] == 0:
                tag = 'LOC'
            else:
                tag = 'LOCORG'
            start = int(split[2][1:-1])
            end = int(split[3][:-1])
            row.append('%s %d %d\n' % (tag, start, end-start+1))
        else:
            split = token.split()
            start = int(split[2][1:-1])
            end = int(split[3][:-1])
            row.append('%s %d %d\n' % (split[0], start, end-start+1))
    with open('./result/' + u[i].name + '.task1', 'w') as f:
        f.writelines(row)

