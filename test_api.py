import numpy as np
import requests
import json

test_input = [
    {"N": 105,
     "P": 14,
     "K": 50,
     "temperature": 26,
     "humidity": 88,
     "ph":6.4,
     "rainfall":59.655
     }
]


payload = json.dumps({"X_test": X_test_list})
payload2 = {'data': 'buy your itunes card'}

headers = {'Content-Type': 'application/json'}
y_predict = requests.post('http://127.0.0.1:3000/predict',headers = headers, json=payload).json()

y_predict = np.array(y_predict)
print(y_predict)
