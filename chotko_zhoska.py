import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder

data = pd.DataFrame({
    "жанр": ["Приключения", "RPG", "Стратегия", "Симулятор", "Гонки", "Гонки", "Шутеры", "Файтинги", "Боевики"],
    "время игры (ч)": [20, 70, 50, 5, 10, 15, 100, 15, 120],
    "оценки пользователей": [70, 35, 50, 90, 85, 80, 30, 75, 25],
    "Тип": ["казуальная", "хардкорная", "казуальная", "казуальная", "казуальная", "казуальная", "хардкорная", "казуальная", "хардкорная"]
})

encoder = OneHotEncoder()
x_brand = encoder.fit_transform(data[["жанр"]]).toarray()
x_price = data[["время игры (ч)", "оценки пользователей"]].values
x = np.hstack([x_brand, x_price])
y = data["Тип"]

clf = DecisionTreeClassifier()
clf.fit(x, y)

new_data = pd.DataFrame({"жанр": ["Шутеры"], "время игры (ч)": [80], "оценки пользователей": [30]})
genre = encoder.transform(new_data[["жанр"]]).toarray()
other = new_data[["время игры (ч)", "оценки пользователей"]].values
sample = np.hstack([genre, other])

result = clf.predict(sample)[0]
print(f"Жанр: {new_data['жанр'][0]} определен как {result}")