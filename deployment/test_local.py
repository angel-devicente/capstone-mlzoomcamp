import requests

url = 'http://localhost:8080/2015-03-31/functions/function/invocations'

data = {'url': 'https://mammography-test.s3.eu-west-2.amazonaws.com/mdb182.jpg'}

result = requests.post(url, json=data).json()
print(result)
