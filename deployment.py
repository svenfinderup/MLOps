import requests
response=requests.get("https://api.github.com/repos/SkafteNicki/dtu_mlops")
print(response.status_code)
if response.status_code == 200:
    print('Success!')
elif response.status_code == 404:
    print('Not Found.')
elif response.status_code == 403:
    print('request rate exceeds the limit for unauthenticated users.')
#print(response.headers)
#data = response.json()
#print("Repository Name:", data['name'])
#print("Full Name:", data['full_name'])
#print("Description:", data['description'])  # Add this if you want to see the description

pload = {'username': 'Olivia', 'password': '123'}
response = requests.post('https://httpbin.org/post', data=pload)
print(response.status_code)
print(response.json())



