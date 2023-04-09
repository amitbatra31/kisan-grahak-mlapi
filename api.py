from flask import Flask, jsonify,request
from flask_restful import Api, Resource, reqparse
import pickle
import numpy as np
import json
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
# from sklearn.externals import joblib
app = Flask(__name__)
api = Api(app)

# Create parser for the payload data
parser = reqparse.RequestParser()
parser.add_argument('data')

# Define how the api will respond to the post requests
class IrisClassifier(Resource):
    def post(self):
        print("Header is", request.headers.get('Content-Type'))
        args = parser.parse_args()
        X = np.array(json.loads(args['data']))
        prediction = model.predict(X)
        return jsonify(prediction.tolist())


api.add_resource(IrisClassifier, '/iris')


@app.route('/predict', methods=['POST'])
def predict():

    if request.method == 'POST':

        print(request.args.get('N'))

        test_d = {
            "N": float(request.args.get('N')),
            "P": float(request.args.get('P')),
            "K": float(request.args.get('K')),
            "temperature": float(request.args.get('temperature')),
            "humidity": float(request.args.get('humidity')),
            "ph": float(request.args.get('ph')),
            "rainfall":float(request.args.get('rainfall'))
        }

        test_input = pd.DataFrame(test_d, index=[0])
        my_prediction = model.predict(test_input)
        return jsonify(my_prediction.tolist())

if __name__ == '__main__':
    # Load model
    # with open('model.pickle', 'rb') as f:
    #     model = pickle.load(f)
    with open('model.pickle', 'rb') as f:
        model = pickle.load(f)
    app.run(debug=True,port=3000)
