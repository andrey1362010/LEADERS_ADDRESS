![Alt text](https://i0.wp.com/neptune.ai/wp-content/uploads/2022/10/When-to-Choose-CatBoost-Over-XGBoost-or-LightGBM-Practical-Guide_13.png?resize=771%2C431&ssl=1)

***Как собрать докер***
-  Собираем докер: ```sudo docker build --tag address_hackaton```
-  Запускаем Docker: ```sudo docker run --publish 8000:5000 --rm address_hackaton```

***Как делать запросы, после того как запустили Docker***
- http://localhost:8000/?query=Санкт-Петербург
- Параметр query должен быть ***URL-Encoded*** (Не как в примере)

***Другая реализация:***
> Первой реализацией был подход через нейронные сети ***LaBSE*** и ***BERT***.
> Данный подход набрал меньший скор и более долгий в обучении и разработке, поэтому оставляем как историю.
> Смотрите файл ***Train_network.ipynb***

Суть подхода в получении вектора для каждого адресса и сравнение со списком готовых строений в файле ***buildings*** по ***Cosine distanse.csv***.
Таким образом не нужно переобучать сеть при добавлении новых адресов в этот файл.

- На вход сети ***Last Hidden State*** из ***BERT*** (Массив embeddings) для каждого адреса
- На выходе новый Embedding 
- Triplet Loss
- Поэлементное сравнение со списком адресов и поиск ***topK***
- Адаптивный ***Hard Sampling***