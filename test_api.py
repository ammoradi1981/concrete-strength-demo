# test_api.py
import requests

url = "http://127.0.0.1:5000/predict"
data = {
    "cement": 350,
    "slag": 50,
    "fly_ash": 0,
    "water": 160,
    "superplasticizer": 5,
    "coarse": 1000,
    "fine": 700,
    "age": 28
}

response = requests.post(url, json=data)
print(response.json())
