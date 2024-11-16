#acá se crea el modelo

"""Build, deploy and access a model using scikit-learn"""

import pickle # crear un archivo

import pandas as pd  # type: ignore - leer la data
from sklearn.linear_model import LinearRegression  # type: ignore

df = pd.read_csv("files\input\house_data.csv", sep=",") #cargo la data

features = df[ #se identifican las diferentes variables, solod e interés
        #estas permiten pronosticar el precio

    [
        "bedrooms",
        "bathrooms",
        "sqft_living",
        "sqft_lot",
        "floors",
        "waterfront",
        "condition",
    ]
]

target = df[["price"]] #Salida - variable a predecir

estimator = LinearRegression() 
estimator.fit(features, target) #ajuste (x,y)
# es necesario guardarlo para poder utilizarlo


with open("homework/house_predictor.pkl", "wb") as file: #crear un archivo, guarda el modelo ya generado
    pickle.dump(estimator, file)