# Импортируем функцию load_iris из библиотеки scikit-learn.
# Она загружает встроенный набор данных "Iris" — классический пример для обучения алгоритмов классификации.

from sklearn.datasets import load_iris
import pandas as pd

# Импортируем библиотеку pandas для работы с табличными данными (DataFrame).
# DataFrame — это как таблица Excel, но управляется через Python.
import pandas as pd

from pprint import pprint

iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target

print(df.head())

# pprint(iris)

