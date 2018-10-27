import sys
sys.path.append('scripts/dialent/task1/')
sys.path.append('scripts/dialent/')
sys.path.append('scripts/')
import util
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import xgboost as xgb

u_dev = util.loadAllStandard('./devset/')
u_test = util.loadAllStandard('./testset//')

def ne_bin(x):
    if x == 'LOC':
        return 0
    elif x == 'LOCORG':
        return 1
    return 2

def get_df(u):
    df = pd.DataFrame(columns=['tokens', 'tokens_id', 'start', 'end', 'ne'])
    for i in range(len(u)):
        res = u[i].makeTokenSets()
        for j in range(len(res)):
            token = res[j].toInlineString()
            split = token.split()
            df = df.append({'tokens': split[4:], 
                            'first_token': str(split[4].split('"')[1]), 
                            'tokens_id': int(split[1]),
                            'ne_bin': ne_bin(str(split[0]))}, ignore_index=True)
    return df

df_dev = get_df(u_dev)
df_test = get_df(u_test)

df_dev = df_dev[df_dev['ne_bin'] < 2]
df_test = df_test[df_test['ne_bin'] < 2]

n = df_dev.shape[0]
vec = TfidfVectorizer(ngram_range=(1, 5), analyzer='char', min_df=2, max_df=0.9, strip_accents='unicode', use_idf=1,
                      smooth_idf=1, sublinear_tf=1)
trn_term_doc = vec.fit_transform(df_dev['first_token'])
test_term_doc = vec.transform(df_test['first_token'])

Y_dev = df_dev['ne_bin']
Y_test = df_test['ne_bin']

best_parameters = {
    'colsample_bytree': 0.4,
    'gamma': 1.1,
    'learning_rate': 0.1,
    'max_depth': 5,
    'min_child_weight': 1,
    'reg_alpha': 0
}

clf = xgb.XGBClassifier(**best_parameters)
clf.fit(trn_term_doc, Y_dev)

predict = clf.predict(test_term_doc)

u = util.loadAllStandard('./testset/')
for i in range(len(u)):
    res = u[i].makeTokenSets()
    row = []
    for j in range(len(res)):
        token = res[j].toInlineString()
        if token[0] == 'L':
            split = token.split()
            if predict[df_test['tokens_id'] == int(split[1])][0] == 0:
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

