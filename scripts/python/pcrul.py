# If you have Python installed

import requests
response = requests.post(
    'http://localhost:6001/v1/generate',
    json={'prompt': 'Hello', 'max_tokens': 10, 'temperature': 0.7, 'device': 'cuda'},
    headers={'Content-Type': 'application/json'},
    proxies={'http': None, 'https': None}
)
print(f'Status: {response.status_code}')
print(f'Response: {response.text}')


# curl --noproxy "*" -X POST http://localhost:6001/v1/generate \
#   -H "Content-Type: application/json" \
#   -d '{"prompt": "Hello", "max_tokens": 10, "temperature": 0.7, "device": "cuda"}'