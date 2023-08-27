from flask import Flask, request
from catboost import CatBoostClassifier
import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
from tqdm import tqdm
import joblib
from sklearn.decomposition import TruncatedSVD

app = Flask(__name__)


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


vector_size = 20
a = np.arange(0, 1.1, 0.25)

tfidf_vectorizer = joblib.load('data/tfidf_vectorizer.joblib')
model_ft = joblib.load('data/model_ft.joblib')
svd = joblib.load('data/svd.joblib')
model = CatBoostClassifier()
model.load_model('data/catboost_model.cbm')

building = pd.read_csv('data/building_20230808.csv')
building = building.drop_duplicates(subset = 'full_address')
building.replace({'nan': np.nan}, inplace = True)
building = building.dropna(subset = ['full_address'])
print(building['is_actual'].value_counts())
# building = building[building['is_actual'] == True]
building.index = range(len(building))
# street = pd.read_csv('train_dataset/additional_data/geonim_20230808.csv')

building.rename(columns={'id': 'target_building_id'}, inplace = True)

building['target_building_id'] = building['target_building_id'].astype(int)

stop_words = set(stopwords.words('russian'))
addresses = building['full_address'].values
ids = building['target_building_id'].values
builds_prep = compare_strings(building['full_address'], a)



def process_address(row):
    ones_ar = np.ones(builds_prep.shape)
    n_top = 10
    test = compare_strings_one(row, a)
    test = test.flatten() * ones_ar
    test_all = np.hstack((test, builds_prep, test - builds_prep))
    add_test_metrics = pd.DataFrame(test_all, columns = [f'emb{i}' for i in range(test_all.shape[1])])
    add_test_metrics['address'] = row
    add_test_metrics['full_address'] = building['full_address']

    predict = model.predict_proba(add_test_metrics)
    ind = np.argsort(predict[:, 1])[::-1]
    top = list(addresses[ind[:n_top]])
    top_ids = list(ids[ind[:n_top]])
    print(top)
    print(top_ids)
    return [(int(bid), bad) for bid, bad in zip(top_ids, top)]

@app.route('/', methods=['GET'])
def process():
    args = request.args
    if "query" not in args: return "query param is required", 500
    query = args.get("query")
    if len(query) < 4: return "query param is too short", 500

    result = process_address(query)

    return {
        "success": True,
        "query:": {
            "address": query
        },
        "result": [
            {"target_building_id": building, "target_address": address} for building, address in result
        ]
    }

if __name__ == '__main__':
    app.run()