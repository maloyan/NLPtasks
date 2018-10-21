import os
import pymorphy2
import numpy as np
import string
from scipy.sparse import csr_matrix
from nltk.tree import Tree
from nltk.util import LazyMap, LazyConcatenation
from nltk.corpus.reader.util import *
from nltk.corpus.reader.api import *
from nltk.stem import WordNetLemmatizer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn import preprocessing
from collections import Counter

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

import warnings

from sklearn.svm import LinearSVC

warnings.filterwarnings('ignore')

TRAINSET_PATH = "./factrueval_trainset.npz"
TESTSET_PATH = "./factrueval_testset.npz"

class Generator:
    def __init__(self,
                 column_types=None,
                 context_len=2,
                 language='ru',
                 number_of_occurences=5,
                 weight_percentage=0.9):

        # Частота, ниже которой лейбл считается "редким" #
        self.NUMBER_OF_OCCURENCES = number_of_occurences

        # Процент веса признаков, который нужно оставить
        self.WEIGHT_PERCENTAGE = weight_percentage  #

        # Информация о подаваемых столбцах (может быть WORD, POS, CHUNK) #
        self._column_types = column_types if column_types is not None else ["WORD"]

        # Длина рассматриваемого контекста (context_len влево и context_len вправо) #
        self._context_len = context_len

        # Анализатор (для POS-тега и начальной формы) #
        self._morph = pymorphy2.MorphAnalyzer()
        self._lemmatizer = WordNetLemmatizer()

        # Язык датасета (определяет используемые модули) #
        self._lang = language

        # OneHotEncoder, хранится после FIT-а #
        self._enc = None

        # ColumnApplier, хранится после FIT-а #
        self._multi_encoder = None

        # Словари распознаваемых слов, хранятся после FIT-а #
        self._counters = []

        # Число столбцов в "сырой" матрице признаков #
        self._number_of_columns = None

        # Индексы столбцов признаков, оставленных после отсева #
        self._columns_to_keep = None

    def fit_transform(self, data, answers, path, clf=ExtraTreesClassifier()):

        # Eсли данные сохранены - просто берем их из файла #
        if os.path.exists(path):
            sparse_features_list = self.load_sparse_csr(path)
            return sparse_features_list

        # Добавляем пустые "слова" в начало и конец (для контекста) #
        data = [["" for i in range(len(self._column_types))] for i in range(self._context_len)] + data
        data = data + [["" for i in range(len(self._column_types))] for i in range(self._context_len)]

        # Находим индексы столбцов в переданных данных #
        word_index = self._column_types.index("WORD")
        if "POS" in self._column_types:
            pos_index = self._column_types.index("POS")
        else:
            pos_index = None
        if "POS" in self._column_types:
            chunk_index = self._column_types.index("CHUNK")
        else:
            chunk_index = None

        # Список признаков (строка == набор признаков для слова из массива data) #
        features_list = []

        # Заполнение массива features_list "сырыми" данными (без отсева) #
        for k in range(len(data) - 2 * self._context_len):
            arr = []
            i = k + self._context_len

            if pos_index is not None:
                pos_arr = [data[i][pos_index]]
                for j in range(1, self._context_len + 1):
                    pos_arr.append(data[i - j][pos_index])
                    pos_arr.append(data[i + j][pos_index])
            else:
                pos_arr = [self.get_pos_tag(data[i][word_index])]
                for j in range(1, self._context_len + 1):
                    pos_arr.append(self.get_pos_tag(data[i - j][word_index]))
                    pos_arr.append(self.get_pos_tag(data[i + j][word_index]))
            arr += pos_arr

            if chunk_index is not None:
                chunk_arr = [data[i][chunk_index]]
                for j in range(1, self._context_len + 1):
                    chunk_arr.append(data[i - j][chunk_index])
                    chunk_arr.append(data[i + j][chunk_index])
                arr += chunk_arr

            capital_arr = [self.get_capital(data[i][word_index])]
            for j in range(1, self._context_len + 1):
                capital_arr.append(self.get_capital(data[i - j][word_index]))
                capital_arr.append(self.get_capital(data[i + j][word_index]))
            arr += capital_arr

            is_punct_arr = [self.get_is_punct(data[i][word_index])]
            for j in range(1, self._context_len + 1):
                is_punct_arr.append(self.get_is_punct(data[i - j][word_index]))
                is_punct_arr.append(self.get_is_punct(data[i + j][word_index]))
            arr += is_punct_arr

            is_number_arr = [self.get_is_number(data[i][word_index])]
            for j in range(1, self._context_len + 1):
                is_number_arr.append(self.get_is_number(data[i - j][word_index]))
                is_number_arr.append(self.get_is_number(data[i + j][word_index]))
            arr += is_number_arr

            initial_arr = [self.get_initial(data[i][word_index])]
            for j in range(1, self._context_len + 1):
                initial_arr.append(self.get_initial(data[i - j][word_index]))
                initial_arr.append(self.get_initial(data[i + j][word_index]))
            arr += initial_arr

            features_list.append(arr)

        # Теперь это массив сырых признаков (в строковом представлении, без отсева) #
        features_list = np.array([np.array(line) for line in features_list])

        # Выкинем из этого массива классы, встретившиеся менее NUMBER_OF_OCCURENCES раз #
        # Посчитаем частоту лейблов в столбце #
        self._number_of_columns = features_list.shape[1]
        for u in range(self._number_of_columns):
            arr = features_list[:, u]
            counter = Counter(arr)
            self._counters.append(counter)

        # Избавимся от редких лейблов (частота < NUMBER_OF_OCC) #
        for y in range(len(features_list)):
            for x in range(self._number_of_columns):
                features_list[y][x] = self.get_feature(x, features_list[y][x])

        # Оставшиеся признаки бинаризуем #
        self._multi_encoder = ColumnApplier(
            dict([(i, preprocessing.LabelEncoder()) for i in range(len(features_list[0]))]))
        features_list = self._multi_encoder.fit(features_list, None).transform(features_list)
        self._enc = preprocessing.OneHotEncoder(dtype=np.bool_, sparse=True)
        self._enc.fit(features_list)
        features_list = self._enc.transform(features_list)

        # Избавляемся от неинформативных признаков (WEIGHT = WEIGHT_PERC * TOTAL_WEIGHT)#
        clf.fit(features_list, answers)
        features_importances = [(i, el) for i, el in enumerate(clf.feature_importances_)]

        features_importances = sorted(features_importances, key=lambda el: -el[1])
        current_weight = 0.0
        self._columns_to_keep = []
        for el in features_importances:
            self._columns_to_keep.append(el[0])
            current_weight += el[1]
            if current_weight > self.WEIGHT_PERCENTAGE:
                break

        features_list = features_list[:, self._columns_to_keep]

        # Сохраняем матрицу в файл #
        self.save_sparse_csr(path, features_list)

        # Возвращаем матрицу #
        return features_list

    def transform(self, data, path):

        # Eсли данные сохранены - просто берем их из файла #
        if os.path.exists(path):
            sparse_features_list = self.load_sparse_csr(path)
            return sparse_features_list

        # Добавляем пустые "слова" в начало и конец (для контекста) #
        data = [["" for i in range(len(self._column_types))] for i in range(self._context_len)] + data
        data = data + [["" for i in range(len(self._column_types))] for i in range(self._context_len)]

        # Находим индексы столбцов в переданных данных #
        word_index = self._column_types.index("WORD")
        if "POS" in self._column_types:
            pos_index = self._column_types.index("POS")
        else:
            pos_index = None
        if "CHUNK" in self._column_types:
            chunk_index = self._column_types.index("CHUNK")
        else:
            chunk_index = None

        # Список признаков (строка == набор признаков для слова из массива data) #
        features_list = []

        # Заполнение массива features_list "сырыми" данными (без отсева) #
        for k in range(len(data) - 2 * self._context_len):
            arr = []
            i = k + self._context_len

            if pos_index is not None:
                pos_arr = [data[i][pos_index]]
                for j in range(1, self._context_len + 1):
                    pos_arr.append(data[i - j][pos_index])
                    pos_arr.append(data[i + j][pos_index])
            else:
                pos_arr = [self.get_pos_tag(data[i][word_index])]
                for j in range(1, self._context_len + 1):
                    pos_arr.append(self.get_pos_tag(data[i - j][word_index]))
                    pos_arr.append(self.get_pos_tag(data[i + j][word_index]))
            arr += pos_arr

            if chunk_index is not None:
                chunk_arr = [data[i][chunk_index]]
                for j in range(1, self._context_len + 1):
                    chunk_arr.append(data[i - j][chunk_index])
                    chunk_arr.append(data[i + j][chunk_index])
                arr += chunk_arr

            capital_arr = [self.get_capital(data[i][word_index])]
            for j in range(1, self._context_len + 1):
                capital_arr.append(self.get_capital(data[i - j][word_index]))
                capital_arr.append(self.get_capital(data[i + j][word_index]))
            arr += capital_arr

            is_punct_arr = [self.get_is_punct(data[i][word_index])]
            for j in range(1, self._context_len + 1):
                is_punct_arr.append(self.get_is_punct(data[i - j][word_index]))
                is_punct_arr.append(self.get_is_punct(data[i + j][word_index]))
            arr += is_punct_arr

            is_number_arr = [self.get_is_number(data[i][word_index])]
            for j in range(1, self._context_len + 1):
                is_number_arr.append(self.get_is_number(data[i - j][word_index]))
                is_number_arr.append(self.get_is_number(data[i + j][word_index]))
            arr += is_number_arr

            initial_arr = [self.get_initial(data[i][word_index])]
            for j in range(1, self._context_len + 1):
                initial_arr.append(self.get_initial(data[i - j][word_index]))
                initial_arr.append(self.get_initial(data[i + j][word_index]))
            arr += initial_arr

            features_list.append(arr)

        # Теперь это массив сырых признаков (в строковом представлении, без отсева) #
        features_list = np.array([np.array(line) for line in features_list])

        # Выкинем из этого массива классы, встретившиеся менее NUMBER_OF_OCCURENCES раз #
        self._number_of_columns = features_list.shape[1]
        for y in range(len(features_list)):
            for x in range(self._number_of_columns):
                features_list[y][x] = self.get_feature(x, features_list[y][x])

        # Оставшиеся признаки бинаризуем #
        features_list = self._multi_encoder.transform(features_list)
        features_list = self._enc.transform(features_list)

        # Избавляемся от неинформативных признаков (WEIGHT = WEIGHT_PERC * TOTAL_WEIGHT)#
        features_list = features_list[:, self._columns_to_keep]

        # Сохраняем матрицу в файл #
        self.save_sparse_csr(path, features_list)

        # Возвращаем матрицу #
        return features_list

    # Заменяет лейбл на "*", если он "редкий" #
    def get_feature(self, f, feature):
        if feature in self._counters[f].keys() and self._counters[f][feature] > self.NUMBER_OF_OCCURENCES:
            return feature
        else:
            return "*"

    # Сохраняет матрицу в файл #
    def save_sparse_csr(self, filename, array):
        np.savez(filename,
                 data=array.data,
                 indices=array.indices,
                 indptr=array.indptr,
                 shape=array.shape)

    # Загружает матрицу из файла #
    def load_sparse_csr(self, filename):
        loader = np.load(filename)
        return csr_matrix((loader['data'],
                           loader['indices'],
                           loader['indptr']),
                          shape=loader['shape'])

    # Возвращает POS-тег слова #
    def get_pos_tag(self, token):
        if self._lang == 'ru':
            pos = self._morph.parse(token)[0].tag.POS
        else:
            pos = None
        if pos is not None:
            return pos
        else:
            return "none"

    # Возвращает тип регистра слова #
    def get_capital(self, token):
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

    # Признак того, является ли слово числом #
    def get_is_number(self, token):
        try:
            complex(token)
        except ValueError:
            return "no"
        return "yes"

    # Возвращает начальную форму слова #
    def get_initial(self, token):
        if self._lang == 'ru':
            initial = self._morph.parse(token)[0].normal_form
        else:
            initial = self._lemmatizer.lemmatize(token)

        if initial is not None:
            return initial
        else:
            return "none"

    # Признак того, является ли слово пунктуацией #
    def get_is_punct(self, token):
        pattern = re.compile("[{}]+$".format(re.escape(string.punctuation)))
        if pattern.match(token):
            return "yes"
        else:
            return "no"


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
class ConllCorpusReaderX(CorpusReader):

    WORDS = 'words'   #: column type for words
    POS = 'pos'       #: column type for part-of-speech tags
    TREE = 'tree'     #: column type for parse trees
    CHUNK = 'chunk'   #: column type for chunk structures
    NE = 'ne'         #: column type for named entities
    SRL = 'srl'       #: column type for semantic role labels
    IGNORE = 'ignore' #: column type for column that should be ignored
    OFFSET = 'offset'
    LEN = 'len'

    #: A list of all column types supported by the conll corpus reader.
    COLUMN_TYPES = (WORDS, POS, TREE, CHUNK, NE, SRL, IGNORE, OFFSET, LEN)

    #/////////////////////////////////////////////////////////////////
    # Constructor
    #/////////////////////////////////////////////////////////////////

    def __init__(self, root, fileids, columntypes,
                 chunk_types=None, root_label='S', pos_in_tree=False,
                 srl_includes_roleset=True, encoding='utf8',
                 tree_class=Tree, tagset=None):
        for columntype in columntypes:
            if columntype not in self.COLUMN_TYPES:
                raise ValueError('Bad column type %r' % columntype)
        if isinstance(chunk_types, string_types):
            chunk_types = [chunk_types]
        self._chunk_types = chunk_types
        self._colmap = dict((c,i) for (i,c) in enumerate(columntypes))
        self._pos_in_tree = pos_in_tree
        self._root_label = root_label # for chunks
        self._srl_includes_roleset = srl_includes_roleset
        self._tree_class = tree_class
        CorpusReader.__init__(self, root, fileids, encoding)
        self._tagset = tagset

    def words(self, fileids=None):
        self._require(self.WORDS)
        return LazyConcatenation(LazyMap(self._get_words, self._grids(fileids)))

    def _grids(self, fileids=None):
        # n.b.: we could cache the object returned here (keyed on
        # fileids), which would let us reuse the same corpus view for
        # different things (eg srl and parse trees).
        return concat([StreamBackedCorpusView(fileid, self._read_grid_block,
                                              encoding=enc)
                       for (fileid, enc) in self.abspaths(fileids, True)])

    def _read_grid_block(self, stream):
        grids = []
        for block in read_blankline_block(stream):
            block = block.strip()
            if not block: continue

            grid = [line.split() for line in block.split('\n')]

            # If there's a docstart row, then discard. ([xx] eventually it
            # would be good to actually use it)
            if grid[0][self._colmap.get('words', 0)] == '-DOCSTART-':
                del grid[0]

            # Check that the grid is consistent.
            for row in grid:
                if len(row) != len(grid[0]):
                    raise ValueError('Inconsistent number of columns:\n%s'
                                     % block)
            grids.append(grid)
        return grids

    def get_ne(self, fileids=None, tagset=None):
        self._require(self.NE)
        def get_ne_inn(grid):
            return self._get_ne(grid, tagset)
        return LazyConcatenation(LazyMap(get_ne_inn, self._grids(fileids)))

    def _get_words(self, grid):
        return self._get_column(grid, self._colmap['words'])

    def _get_ne(self, grid, tagset=None):
        return list(zip(self._get_column(grid, self._colmap['words']),
                        self._get_column(grid, self._colmap['ne'])))

    def _require(self, *columntypes):
        for columntype in columntypes:
            if columntype not in self._colmap:
                raise ValueError('This corpus does not contain a %s '
                                 'column.' % columntype)
    @staticmethod
    def _get_column(grid, column_index):
        return [grid[i][column_index] for i in range(len(grid))]

# Избавляет данные от случаев O : O #
def clean(Y_pred, Y_test):
    Y_pred = np.array(Y_pred)
    Y_test = np.array(Y_test)

    Y_pred_i = np.array([Y_pred != 'O'])
    Y_test_i = np.array([Y_test != 'O'])

    indexes = (Y_pred_i | Y_test_i).reshape(Y_pred.shape)

    Y_pred_fixed = Y_pred[indexes]
    Y_test_fixed = Y_test[indexes]
    return Y_pred_fixed, Y_test_fixed
def prepare(dataset):
    factrueval_dev_tokens = dict()
    factrueval_dev_tokens_list = []
    factrueval_dev_spans = dict()
    factrueval_dev_objects = dict()
    for file in os.listdir('./'+ dataset + '/'):
        if file.endswith('tokens'):
            with open('./' + dataset + '/' + file, 'r+', encoding='utf-8') as file_obj:
                lines = file_obj.readlines()
                tokens = [line.rstrip().split() for line in lines if line.rstrip().split() != []]
                for token in tokens:
                    factrueval_dev_tokens[int(token[0])] = token[1:]
                tokens = [line.rstrip().split() for line in lines]
                for token in tokens:
                    factrueval_dev_tokens_list.append(token)

        if file.endswith('spans'):
            with open('./' + dataset + '/' + file, 'r+', encoding='utf-8') as file_obj:
                spans = [line.rstrip().split() for line in file_obj.readlines() if line.rstrip().split() != []]
                for span in spans:
                    factrueval_dev_spans[span[0]] = span[1:]

        if file.endswith('objects'):
            with open('./' + dataset + '/' + file, 'r+', encoding='utf-8') as file_obj:
                objects = [line.rstrip().split('#')[0].split() for line in file_obj.readlines() if line.rstrip().split() != []]
                for obj in objects:
                    factrueval_dev_objects[obj[0]] = obj[1:]

    all_ne = []
    for key, value in factrueval_dev_objects.items():
        spans = value[1:]
        ne = value[0]
        all_tokens = []
        for span in spans:
            span_obj = factrueval_dev_spans[span]
            token = int(span_obj[3])
            num_of_tokens = int(span_obj[4])
            for i in range(num_of_tokens):
                all_tokens.append(token + i)
        all_ne.append([ne, sorted(all_tokens)])

    for ne_tokens in all_ne:
        ne = ne_tokens[0]
        token = ne_tokens[1]
        for i in range(len(token)):
            if token[i] in factrueval_dev_tokens.keys():
                if len(token) == 1:
                    factrueval_dev_tokens[token[i]].append("S-" + ne)
                elif (i == 0 and token[i + 1] - token[i] > 1) or (i == len(token) - 1 and token[i] - token[i - 1] > 1) or (token[i] - token[i - 1] > 1 and token[i + 1] - token[i] > 1):
                    factrueval_dev_tokens[token[i]].append("S-" + ne)
                elif (i == 0  and token[i + 1] - token[i] == 1) or (i != len(token) - 1 and token[i] - token[i - 1] > 1 and token[i + 1] - token[i] == 1):
                    factrueval_dev_tokens[token[i]].append("B-" + ne)
                elif (i == len(token) - 1 and token[i] - token[i - 1] == 1) or (i != 0 and token[i] - token[i - 1] == 1 and token[i + 1] - token[i] > 1):
                    factrueval_dev_tokens[token[i]].append("E-" + ne)
                else:
                    factrueval_dev_tokens[token[i]].append("I-" + ne)

    for i in range(len(factrueval_dev_tokens_list)):
        if factrueval_dev_tokens_list[i] == []:
            continue
        number_of_token = factrueval_dev_tokens_list[i][0]
        if int(number_of_token) in factrueval_dev_tokens.keys() and len(factrueval_dev_tokens[int(number_of_token)]) >= 4:
            ne = factrueval_dev_tokens[int(number_of_token)][3]
            factrueval_dev_tokens_list[i].append(ne)
        else:
            factrueval_dev_tokens_list[i].append("O")

    final = []
    for el in factrueval_dev_tokens_list:
        if el == []:
            final.append(el)
        else:
            final.append([el[3], el[1], el[2], el[4]])
    return final

def dataSetFile(path, dataset):
    with open(path, 'w+', encoding='utf-8') as file:
        for line in dataset:
            if line == []:
                file.write("\n")
            else:
                file.write("{} {} {} {}\n".format(*line))

devset = prepare('devset')
dataSetFile('./devset.txt', devset)
testset = prepare('testset')
dataSetFile('./testset.txt', testset)

factrueval_devset = ConllCorpusReaderX('./', fileids='devset.txt', columntypes=['words', 'offset', 'len', 'ne'])
factrueval_testset = ConllCorpusReaderX('./', fileids='testset.txt', columntypes=['words', 'offset', 'len', 'ne'])

gen = Generator(column_types=['WORD'], context_len=2)

Y_train = [el[1] for el in factrueval_devset.get_ne()]
Y_test = [el[1] for el in factrueval_testset.get_ne()]

X_train = gen.fit_transform([[el] for el in factrueval_devset.words()], Y_train, path=TRAINSET_PATH)
X_test = gen.transform([[el] for el in factrueval_testset.words()], path=TESTSET_PATH)

def run_baseline(clf=LogisticRegression()):
    clf.fit(X_train, Y_train)
    Y_pred = clf.predict(X_test)
    Y_pred_c, Y_test_c = clean(Y_pred, Y_test)

    def get_el(el):
        if el == "O":
            return el
        else:
            return el[2:]

    Y_pred_c_light = [get_el(el) for el in Y_pred_c]
    Y_test_c_light = [get_el(el) for el in Y_test_c]

    # Strict evaluation #

    print("")
    print("# Strict evaluation #")
    counter = Counter(Y_test_c)
    labels = list(counter.keys())
    labels.remove("O")
    results = f1_score(Y_test_c, Y_pred_c, average=None, labels=labels)
    for a, b in zip(labels, results):
        print('F1 for {} == {}, with {} entities'.format(a, b, counter[a]))

    print("Weighted Score:", f1_score(Y_test_c, Y_pred_c, average="weighted", labels=list(counter.keys())))

    # Not strict evaluation #

    print("")
    print("# Not strict evaluation #")
    light_counter = Counter(Y_test_c_light)
    light_labels = list(light_counter.keys())
    light_labels.remove("O")
    print(light_counter)
    light_results = f1_score(Y_test_c_light, Y_pred_c_light, average=None, labels=light_labels)
    for a, b in zip(light_labels, light_results):
        print('F1 for {} == {}, with {} entities'.format(a, b, light_counter[a]))

    print("Weighted Score:", f1_score(Y_test_c_light, Y_pred_c_light, average="weighted", labels=light_labels))
run_baseline()
run_baseline(RandomForestClassifier())
run_baseline(LinearSVC())