import requests


cookies = requests.request("POST", "http://localhost:20202/login", data={"username": "joe", "password": "Monkey$20"}).cookies
response = requests.request("GET", "http://localhost:20202/_shutdown", cookies=cookies)
print(response.content)
