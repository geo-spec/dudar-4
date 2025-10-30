# Импортируем функцию load_iris из библиотеки scikit-learn.
# Она загружает встроенный набор данных "Iris" — классический пример для обучения алгоритмов классификации.

from sklearn.datasets import load_iris
import pandas as pd

# Импортируем библиотеку pandas для работы с табличными данными (DataFrame).
# DataFrame — это как таблица Excel, но управляется через Python.
import pandas as pd

from pprint import pprint

# Загружаем набор данных Iris и сохраняем его в переменную iris.
# Функция возвращает объект типа Bunch (почти как словарь),
# в котором хранятся: данные, названия столбцов, названия классов и описание.

iris = load_iris()

# Создаем объект DataFrame (таблицу) из данных iris.
# Аргумент data=iris.data — это сами числовые данные (длина и ширина лепестков и чашелистиков).
# Аргумент columns=iris.feature_names — это список названий столбцов, чтобы таблица имела читаемые заголовки.

df = pd.DataFrame(data=iris.data, columns=iris.feature_names)

# Добавляем новый столбец 'target' в DataFrame.
# В нем хранятся числовые метки классов (0, 1, 2),
# которые соответствуют трем видам ирисов: setosa, versicolor, virginica.

df['target'] = iris.target


# Выводим первые 5 строк таблицы на экран, чтобы убедиться, что данные загружены правильно.
# Это быстрый способ "взглянуть" на структуру таблицы и убедиться, что всё в порядке.



# print(df.head())

from sklearn.model_selection import train_test_split
X = df[iris.feature_names]
y = df['target']

# X_train, X_test - данные на которых будем обучать модель

# y_train, y_test - тестовые данные для проверки модели (как хорошо модель научилась предсказывать)


X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42
)
# test_size=0.2 означает, что 20% данных уйдет на тестирование модели, а 80% останется для обучения.
# random_state=42 — это "зерно" для генератора случайных чисел (чтобы разбиение было воспроизводимым).
# если не указывать random_state, то при каждом запуске кода разбиение будет разным.

pprint(X_train[:5])
pprint(y_train[:5])

# [:5] - слайсинг, берем первые 5 элементов из массива






# pprint(iris)

