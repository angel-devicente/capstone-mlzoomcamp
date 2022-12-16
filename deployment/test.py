import requests

#url = 'http://localhost:8080/2015-03-31/functions/function/invocations'

url = 'https://ad1p7rqhyj.execute-api.eu-west-2.amazonaws.com/test/predict'
data = {'url': 'https://i.ibb.co/ZXs9SJN/mdb182.jpg'}

result = requests.post(url, json=data).json()
print(result)
