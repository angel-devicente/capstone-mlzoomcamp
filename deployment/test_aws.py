import requests

url = 'https://ad1p7rqhyj.execute-api.eu-west-2.amazonaws.com/test/predict'

data = {'url': 'https://mammography-test.s3.eu-west-2.amazonaws.com/mdb182.jpg'}

result = requests.post(url, json=data).json()
print(result)
