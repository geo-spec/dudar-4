import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# Подготовим данные
data = {
	'age': [25, 45, 35, 22, 55, 33, 27, 40],
	'income': [50000, 90000, 65000, 40000, 120000, 70000, 48000, 80000],
	'purchased': [0, 1, 1, 0, 1, 1, 0, 1]
}

df = pd.DataFrame(data)

# Разделим данные на признаки и цель
X = df[['age', 'income']]
y = df['purchased']

# Разделим выборку
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Обучим модель
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

# Проверим точность
accuracy = model.score(X_test, y_test)
print(f"Точность модели: {accuracy * 100:.2f}%")

# Предсказание для нового пользователя
new_user = [[30, 60000]]
prediction = model.predict(new_user)
print("Купит ли продукт:", "Да" if prediction[0] == 1 else "Нет")