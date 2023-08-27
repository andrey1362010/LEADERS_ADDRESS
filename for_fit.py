try:
    from IPython import get_ipython
    get_ipython().magic('clear')
    get_ipython().magic('reset -f')
except:
    pass

for element in dir():
    if element[0:2] != "__":
        del globals()[element]
del element 

import sys, gc
while gc.collect():
    gc.collect()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox
import seaborn as sns
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'qt')
from sklearn import metrics
from catboost import Pool, CatBoostClassifier
import random
from tqdm import tqdm

import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import FastText
from sklearn.model_selection import KFold
import joblib
from sklearn.decomposition import TruncatedSVD

def preprocess_string(input_string):
    cleaned_string = input_string.replace('.', ' ').replace('-', ' ').replace(',', ' ').lower()
    
    # Токенизация
    tokens = cleaned_string.split()
    
    # Удаление стоп-слов до лемматизации
    filtered_tokens = [token for token in tokens if token not in stop_words]
    
    # Собираем предобработанный текст обратно
    preprocessed_string = ' '.join(filtered_tokens)
    
    return preprocessed_string


    
def compare_strings(doc, a):
    q1 = np.zeros((len(doc), vector_size * len(a)))
    # Применяем предварительную обработку
    pss = []
    numbers_ar = np.zeros((len(doc), 3))
    for i, string in tqdm(enumerate(doc), total = len(doc)):
        ps = preprocess_string(string)
        pss.append(ps)
        try:
            embedding = model_ft.wv[ps.split()]  # Получаем векторное представление адреса
            q1[i, :] = np.quantile(embedding, a, axis = 0).flatten()
        except: pass
        trim_str = string[6:]
        
        try:
            numbers = np.array([int(number) for number in re.findall(r'-?\d+(?:\.\d+)?', trim_str)])
            if len(numbers) < 3:
                numbers_ar[i, :len(numbers)] = numbers
            else:
                numbers_ar[i, :] = numbers[:3]
        except:
            pass        
    
            
    tfidf_1 = tfidf_vectorizer.transform(pss)
    q = svd.transform(tfidf_1)
    return np.hstack((q1, q, numbers_ar))


def compare_strings_one(string, a):
    ps = preprocess_string(string)
    q1 = np.zeros((1, vector_size * len(a)))
    try:
        embedding = model_ft.wv[ps.split()]
        q1[:] = np.quantile(embedding, a, axis = 0).flatten()
    except: pass
    
    trim_str = string[6:]
    numbers_ar = np.zeros((1, 3))
    try:
        numbers = np.array([int(number) for number in re.findall(r'-?\d+(?:\.\d+)?', trim_str)])
        if len(numbers) < 3:
            numbers_ar[0, :len(numbers)] = numbers
        else:
            numbers_ar[0, :] = numbers[:3]
    except:
        pass
        
    tfidf_1 = tfidf_vectorizer.transform([ps])
    q = svd.transform(tfidf_1)
    
    return np.hstack((q1.reshape(1, -1), q.reshape(1, -1), numbers_ar.reshape(1, -1)))



df = pd.DataFrame()
for i in [1, 2, 3, 4, 5]:
    df = pd.concat([df, pd.read_csv(f'train_dataset/datasets/dataset_{i}.csv')])

df = df.dropna()

# area = pd.read_csv('train_dataset/additional_data/area_20230808.csv')
building = pd.read_csv('train_dataset/additional_data/building_20230808.csv')
building = building.drop_duplicates(subset = 'full_address')
building.replace({'nan': np.nan}, inplace = True)
building = building.dropna(subset = ['full_address'])
print(building['is_actual'].value_counts())
# building = building[building['is_actual'] == True]
building.index = range(len(building))
# street = pd.read_csv('train_dataset/additional_data/geonim_20230808.csv')

building.rename(columns={'id': 'target_building_id'}, inplace = True)

building['target_building_id'] = building['target_building_id'].astype(int)
df['target_building_id'] = df['target_building_id'].astype(int)

df = df.merge(building[['target_building_id', 'full_address']], on = 'target_building_id', how = 'left')
df_dd = df.drop_duplicates(subset = ['address', 'full_address'])

prefix = pd.read_csv('train_dataset/additional_data/prefix_20230808.csv')
prefix = prefix.dropna(subset = 'name')

num_rand = 3
num_rand_best = 5

          
HARD_SAMPLES = {}
def add_samples(building_ids):
    if(len(building_ids) < 2): return
    if(len(building_ids) > 200): return
    for building_id_1 in building_ids:
        for building_id_2 in building_ids:
            if building_id_1 != building_id_2:
                if building_id_1 not in HARD_SAMPLES: HARD_SAMPLES[building_id_1] = set()
                HARD_SAMPLES[building_id_1].add(building_id_2)
                    
for building_ids in building.groupby('prefix_id')['target_building_id'].apply(list).to_dict().values():
    add_samples(building_ids)
# for building_ids in building.groupby('district_id')['target_building_id'].apply(list).to_dict().values():
#     add_samples(building_ids)
for building_ids in building.groupby('house')['target_building_id'].apply(list).to_dict().values():
    add_samples(building_ids)
    
new_rows = []
list_adr = list(building['full_address'])
dict_adr = building[['target_building_id', 'full_address']].set_index('target_building_id').to_dict()['full_address']
sum_of_best = 0
for _, row in tqdm(df_dd.iterrows(), total = len(df_dd)):
    random_targets = random.sample(list_adr, num_rand)
    new_rows.append({'address': row['address'], 'full_address': row['full_address'], 'class': 1})
    for target in random_targets:
        new_rows.append({'address': row['address'], 'full_address': target, 'class': 0})
    try:
        best_id = list(HARD_SAMPLES[row['target_building_id']])
        if len(best_id) > num_rand_best:
            random_targets = random.sample(best_id, num_rand_best)
        else:
            random_targets = best_id
        
        for target in random_targets:
            new_rows.append({'address': row['address'], 'full_address': dict_adr[target], 'class': 0})            
            sum_of_best += 1
    except: pass

list_adr = list(building['full_address'])
short_adr = list(building['short_address'])
for i, adr in enumerate(tqdm(list_adr, total = len(list_adr))):
    new_rows.append({'address': short_adr[i], 'full_address': adr, 'class': 1})
    adr_list = adr.split(',')
    for i, el in enumerate(adr_list):
        if 'дом' in el:
            break
    adr_join = ','.join(adr_list[i - 1: i + 1])
    new_rows.append({'address': adr_join, 'full_address': adr, 'class': 1}) 
    adr_join = ','.join(adr_list[i - 2: i + 1])
    new_rows.append({'address': adr_join, 'full_address': adr, 'class': 1})
    adr_join = ','.join(adr_list[i - 2: i + 1])
    adr_join = adr_join.replace('дом ', 'д.')
    adr_join = adr_join.replace('улица', 'ул.')
    adr_join = adr_join.replace('проспект', 'пр-кт')
    adr_join = adr_join.replace('бульвар', 'б-р') 
    adr_join = adr_join.replace('аллея', 'ал.')
    adr_join = adr_join.replace('дорога', 'дор.')
    adr_join = adr_join.replace('площадь', 'пл.')
    adr_join = adr_join.replace('набережная реки', 'наб.')
    adr_join = adr_join.replace('набережная', 'наб.')
    new_rows.append({'address': adr_join, 'full_address': adr, 'class': 1})
    adr_join = ','.join(adr_list[i - 1: i + 1])
    adr_join = adr_join.replace('дом ', 'д.')
    adr_join = adr_join.replace('улица', 'ул.')
    adr_join = adr_join.replace('проспект', 'пр-кт')
    adr_join = adr_join.replace('бульвар', 'б-р') 
    adr_join = adr_join.replace('аллея', 'ал.')
    adr_join = adr_join.replace('дорога', 'дор.')
    adr_join = adr_join.replace('площадь', 'пл.')
    adr_join = adr_join.replace('набережная реки', 'наб.')
    adr_join = adr_join.replace('набережная', 'наб.')
    new_rows.append({'address': adr_join, 'full_address': adr, 'class': 1})
    adr_join = ','.join(adr_list[i - 1: i + 2])
    adr_join = adr_join.replace('дом ', 'д.')
    adr_join = adr_join.replace('улица', 'ул.')
    adr_join = adr_join.replace('проспект', 'пр-кт')
    adr_join = adr_join.replace('бульвар', 'б-р') 
    adr_join = adr_join.replace('аллея', 'ал.')
    adr_join = adr_join.replace('дорога', 'дор.')
    adr_join = adr_join.replace('площадь', 'пл.')
    adr_join = adr_join.replace('набережная реки', 'наб.')
    adr_join = adr_join.replace('набережная', 'наб.')
    new_rows.append({'address': adr_join, 'full_address': adr, 'class': 1})
    

new_df = pd.DataFrame(new_rows)
new_df.replace({'nan': np.nan}, inplace = True)
new_df = new_df.dropna(subset = ['full_address'])
new_df.index = range(len(new_df))


stop_words = set(stopwords.words('russian'))

data = list(building['short_address']) + list(building['full_address']) + list(df_dd['address'])
# Токенизация данных
tokenized_data = [preprocess_string(text) for text in data]

tfidf_vectorizer = TfidfVectorizer(max_features = 1000)
tfidf_matrix = tfidf_vectorizer.fit_transform(tokenized_data)


# Применение SVD для разложения TF-IDF матрицы
n_components = 10  # Количество компонент
svd = TruncatedSVD(n_components=n_components)
svd.fit(tfidf_matrix)

vector_size = 20
model_ft = FastText(tokenized_data, vector_size=vector_size, window=5, min_count=1, sg=1)

joblib.dump(tfidf_vectorizer, 'tfidf_vectorizer.joblib')
joblib.dump(model_ft, 'model_ft.joblib')
joblib.dump(svd, 'svd.joblib')

a = np.arange(0, 1.1, 0.25)

emb_adr = compare_strings(new_df['address'], a)
emb_tar = compare_strings(new_df['full_address'], a)
emb_all = np.hstack((emb_adr, emb_tar, emb_adr - emb_tar))
add_metrics = pd.DataFrame(emb_all, columns = [f'emb{i}' for i in range(emb_all.shape[1])])

X = pd.concat([add_metrics, new_df[['address', 'full_address']]], axis = 1)
y = new_df['class']

folds = 5
cv = KFold(n_splits = folds, shuffle=True, random_state=42)

params = [
    {'loss_function': 'Logloss', 'boosting_type': 'Plain', 'bootstrap_type': 'Bernoulli', 'depth': 4, 'colsample_bylevel': 0.14344097640056408, 'l2_leaf_reg': 1, 'iterations': 7000, 'subsample': 0.9314298834061376},
    ]

models = []

for param in params:
    param['learning_rate'] = 0.5
    param['verbose'] = 10
    param['random_state'] = 42
    models.append(CatBoostClassifier(**param))

# pred = np.zeros((X.shape[0], 2, len(models)))
# for i, model in enumerate(models):   
#     for train_index, test_index in tqdm(cv.split(X)):
#         # model.fit(X.iloc[train_index], y.iloc[train_index])
        
#         train_pool = Pool(
#             X.iloc[train_index], 
#             y.iloc[train_index], 
#             text_features=['address', 'full_address'],
#         )
#         val_pool = Pool(
#             X.iloc[test_index], 
#             y.iloc[test_index], 
#             text_features=['address', 'full_address'],
#         )
#         model.fit(train_pool, eval_set=val_pool)
#         pred[test_index, :, i] += model.predict_proba(X.iloc[test_index])
# pred = np.mean(pred, axis = 2)
# print(metrics.roc_auc_score(y, pred[:, 1]))

# pred_r = pred[:, 1].reshape(-1, 6)
# pred_r /= np.sum(pred_r, axis = 1).reshape(-1, 1)

model = models[0]
train_pool = Pool(
    X, 
    y, 
    text_features=['address', 'full_address']
    )

model.fit(train_pool)
model.save_model('catboost_model.cbm')


addresses = building['full_address'].values
builds_prep = compare_strings(building['full_address'], a)

ones_ar = np.ones(builds_prep.shape)
top = []
scores = []
df_test = pd.read_csv('test_example.csv', on_bad_lines='skip', sep = ';')
n_top = 10
for row in tqdm(list(df_test['address']), total = len(df_test)):
    test = compare_strings_one(row, a)
    test = test.flatten() * ones_ar
    test_all = np.hstack((test, builds_prep, test - builds_prep))
    add_test_metrics = pd.DataFrame(test_all, columns = [f'emb{i}' for i in range(test_all.shape[1])])
    add_test_metrics['address'] = row
    add_test_metrics['full_address'] = building['full_address']    
    
    predict = model.predict_proba(add_test_metrics)    
    ind = np.argsort(predict[:, 1])[::-1]
    top.append(list(addresses[ind[:n_top]]))
    scores.append(predict[[ind[:n_top]], 1])
    

df_test = df_test.merge(building[['target_building_id', 'full_address']], on = 'target_building_id', how = 'left')
score = np.zeros(len(df_test))
for i in tqdm(range(len(df_test)), total = len(df_test)):
    for j in range(n_top):
        if top[i][j] == df_test.loc[i, 'full_address']:
            score[i] = 1

print(np.mean(score))    

# from natasha import (
#     Segmenter,
#     MorphVocab,
#     LOC,
#     AddrExtractor    
# )
# segmenter = Segmenter()
# morph_vocab = MorphVocab()
# addr_extractor = AddrExtractor(morph_vocab)
# #Текст, содержащий адреса
# text = df_test.loc[292, 'address']
# # text = building.loc[i, 'full_address']
# #Извлечение адреса из текста
# matches = addr_extractor(text)
# facts = [i.fact.as_json for i in matches]
# #Цикл для вывода адреса в удобной форме

# for i in range(len(facts)):
#      tmp = list(facts[i].values())
#      print(tmp[0])
#      # print(tmp[1], tmp[0])
     



import telebot
import optuna


bot = telebot.TeleBot('6130201911:AAHrpW5qu7DmHbS-2nf5gHPAN7vxLlZYR7o')
bot.send_message(312849799, 'new_start')

bot.send_message(312849799, str(np.mean(score)))
raise Exception
optuna.logging.set_verbosity(optuna.logging.WARN)

def logging_callback(study, frozen_trial):
    previous_best_value = study.user_attrs.get("previous_best_value", None)
    if previous_best_value != study.best_value:
        study.set_user_attr("previous_best_value", study.best_value)
        mess = "Trial {} value: {} parameters: {}. ".format(
        frozen_trial.number,
        frozen_trial.value,
        frozen_trial.params)      
        bot.send_message(312849799, mess)
        print(mess)
   
   
def objective_cb(trial):
    param = {
            'loss_function': trial.suggest_categorical('loss_function', ['Logloss', 'CrossEntropy']),
            "boosting_type": trial.suggest_categorical("boosting_type", ["Ordered", "Plain"]),
            "bootstrap_type": trial.suggest_categorical("bootstrap_type", ["Bayesian", "Bernoulli", "MVS"]),
            'depth': trial.suggest_int("depth", 2, 6),            
            'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.01, 0.5, log=True),
            'l2_leaf_reg': trial.suggest_int("l2_leaf_reg", 1, 1000, log=True),
            'learning_rate': 0.5,
            'iterations': trial.suggest_int("iterations", 25, 10000, log=True),
            'verbose': False,
            'random_state': 42,
            
        }

    if param["bootstrap_type"] == "Bayesian":
        param["bagging_temperature"] = trial.suggest_float("bagging_temperature", 0, 10)
    elif param["bootstrap_type"] == "Bernoulli":
        param["subsample"] = trial.suggest_float("subsample", 0.1, 1, log=True)
    
    model = CatBoostClassifier(**param)    
    
    train_pool = Pool(
        X, 
        y, 
        text_features=['address', 'full_address']
        )

    model.fit(train_pool)
    
    top = []
    scores = []
    for row in list(df_test['address']):
        test = compare_strings_one(row, a)
        test = test.flatten() * ones_ar
        test_all = np.hstack((test, builds_prep, test - builds_prep))
        add_test_metrics = pd.DataFrame(test_all, columns = [f'emb{i}' for i in range(test_all.shape[1])])
        add_test_metrics['address'] = row
        add_test_metrics['full_address'] = building['full_address']    
        
        predict = model.predict_proba(add_test_metrics)    
        ind = np.argsort(predict[:, 1])[::-1]
        top.append(list(addresses[ind[:n_top]]))
        scores.append(predict[[ind[:n_top]], 1])
    
    score = np.zeros(len(df_test))
    for i in range(len(df_test)):
        for j in range(n_top):
            if top[i][j] == df_test.loc[i, 'full_address']:
                score[i] = 1               
            
    return np.mean(score)
    
    

sampler = optuna.samplers.TPESampler()
pruner = optuna.pruners.MedianPruner(n_startup_trials=2, n_warmup_steps=5, interval_steps=3)
study = optuna.create_study(sampler = sampler, direction = "maximize")
study.optimize(objective_cb, n_trials = 1000, callbacks=[logging_callback], n_jobs = 20)



# def objective_lgb(trial):
#     params = {
#         'max_depth': trial.suggest_int("max_depth", 2, 30, log=True),        
#         'num_iterations': trial.suggest_int("num_iterations", 30, 2000, log=True),
#         'boosting_type': trial.suggest_categorical('boosting_type', ['gbdt', 'dart']),        
#         'num_leaves': trial.suggest_int("num_leaves", 2, 20, log=True),
#         'subsample': trial.suggest_float("subsample", 0.1, 1),
#         'feature_fraction': trial.suggest_float("feature_fraction", 0.1, 1),
#         'bagging_fraction': trial.suggest_uniform('bagging_fraction', 0.4, 1.0),
#         'lambda_l1': trial.suggest_float("lambda_l1", 1e-3, 50, log=True),
#         'lambda_l2': trial.suggest_float("lambda_l2", 1e-3, 50, log=True),
#         'max_bin': trial.suggest_int("max_bin", 10, 200, log=True),        
#         'min_child_samples': trial.suggest_int('min_child_samples', 1, 100),
#         'learning_rate': 0.5,
#         'verbose': -1,
#         'seed': 42
#         }
#     model = LGBMClassifier(**params)
        
    
#     pred = np.zeros((X.shape[0], 2))
   
#     for train_index, test_index in cv.split(X):
#         model.fit(X[train_index], y[train_index])
#         pred[test_index] += model.predict_proba(X[test_index])

#     return metrics.roc_auc_score(y, pred[:, 1])
    
    

# sampler = optuna.samplers.TPESampler()
# pruner = optuna.pruners.MedianPruner(n_startup_trials=2, n_warmup_steps=5, interval_steps=3)
# study = optuna.create_study(sampler = sampler, direction = "maximize")
# study.optimize(objective_cb, n_trials = 1000, callbacks=[logging_callback], n_jobs = 20)


