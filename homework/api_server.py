#
# Usage from command line:
# curl http://127.0.0.1:5000 -X POST -H "Content-Type: application/json" \
# -d '{"bathrooms": "2", "bedrooms": "3", "sqft_living": "1800", \ 
# "sqft_lot": "2200", "floors": "1", "waterfront": "1", "condition": "3"}'
#

#windows
#curl http://127.0.0.1:5000 -X POST -H "Content-Type: application/json" 
# -d "{\"bathrooms\": \"2\", \"bedrooms\": \"3\", \"sqft_living\": \"1800\", 
# \"sqft_lot\": \"2200\", \"floors\": \"1\", \"waterfront\": \"1\", \"condition\": \"3\"}"

import pickle

import pandas as pd  # type: ignore
from flask import Flask, request  # type: ignore

app = Flask(__name__)
app.config["SECRET_KEY"] = "you-will-never-guess"

#lista con los campos
FEATURES = [
    "bedrooms",
    "bathrooms",
    "sqft_living",
    "sqft_lot",
    "floors",
    "waterfront",
    "condition",
]

#aplicacióm
@app.route("/", methods=["POST"])
def index():
    """API function"""

    args = request.json #diccionario en texto
    filt_args = {key: [int(args[key])] for key in FEATURES} #diccionario que tiene listas
    #para cade clave en features, la clave es tal y el valor de la clave en entero
    df = pd.DataFrame.from_dict(filt_args)

    with open("homework/house_predictor.pkl", "rb") as file:
        loaded_model = pickle.load(file)

    prediction = loaded_model.predict(df) #toma el df y registra para cada datos del df 
    #la predicción 

    return str(prediction[0][0]) #un solo valor 

#correrlo en terminal con python homework\api_server.py

if __name__ == "__main__":
    app.run(debug=True)