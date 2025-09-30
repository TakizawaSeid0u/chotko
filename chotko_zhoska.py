import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder

data = pd.DataFrame({
    "категория": ["одежда", "техника", "еда", "одежда", "техника", "еда", "одежда", "техника", "еда"],
    "стоимость": [25, 70, 10, 30, 80, 5, 15, 105, 8],
    "оценки пользователей": [70, 35, 50, 90, 85, 80, 30, 75, 25],
    "Тип": ["казуальная", "хардкорная", "казуальная", "казуальная", "казуальная", "казуальная", "хардкорная", "казуальная", "хардкорная"]
})

encoder = OneHotEncoder()
x_brand = encoder.fit_transform(data[["категория"]]).toarray()
x_price = data[["стоимость"]].values
x = np.hstack([x_brand, x_price])
y = data["Тип"]

clf = DecisionTreeClassifier()
clf.fit(x, y)

new_data = pd.DataFrame({"категория": ["одежда"], "стоимость": [80], "оценки пользователей": [30]})
genre = encoder.transform(new_data[["категория"]]).toarray()
other = new_data[["стоимость", "оценки пользователей"]].values
sample = np.hstack([genre, other])

result = clf.predict(sample)[0]
print(f"категория: {new_data['категория'][0]} определен как {result}") #ffffffff