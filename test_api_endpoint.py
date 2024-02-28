import requests

for model in ["lr", "rf", "svm"]:
    url = f"http://localhost:8000/api/{model}"
    data = {"message": "another data"}
    response = requests.post(url, data=data)
    print(model)
    print(response.status_code)
    print(response.json())
