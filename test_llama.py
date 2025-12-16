import requests

url = "http://localhost:1234/v1/chat/completions"

payload = {
    "model": "llama",
    "messages": [
        {"role": "system", "content": "Kamu adalah asisten AI."},
        {"role": "user", "content": "Halo, jelaskan singkat apa itu LLM"}
    ],
    "temperature": 0.7
}

res = requests.post(url, json=payload)

print("STATUS:", res.status_code)
print(res.json())
