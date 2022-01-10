import pickle
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import RobustScaler
import pandas as pd


class Predict:
    def __init__(self):
        self.cd = os.getcwd()
        df = pd.read_csv(self.cd + "\\vgsales.csv")
        self.Platformle = LabelEncoder()
        self.Genrele = LabelEncoder()
        self.Platformle.fit_transform(df["Platform"])
        df["Platform"] = self.Platformle.transform(df["Platform"])
        self.Genrele.fit_transform(df["Genre"])
        df["Genre"] = self.Genrele.transform(df["Genre"])
        X = df[
            ["Platform", "Genre", "NA_Sales", "EU_Sales", "JP_Sales", "Other_Sales"]
        ].values
        self.ro = RobustScaler()
        self.ro.fit_transform(X)

    def pred(self, data):
        [data[1]] = self.Platformle.transform([data[1]])
        [data[2]] = self.Genrele.transform([data[2]])
        loaded_model = pickle.load(open(self.cd + "\\" + data[0] + ".sav", "rb"))
        data.pop(0)
        test = self.ro.transform([data])
        result = loaded_model.predict(test)
        return result
