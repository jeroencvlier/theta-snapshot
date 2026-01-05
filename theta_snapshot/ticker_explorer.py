import httpx

BASE_URL = "http://127.0.0.1:25500/v2"
url = f"{BASE_URL}/list/roots"  # Correct endpoint for listing option roots/symbols

# Params: sec=option (required), stype= (optional: s for stock, i for index, etc.)
params = {"sec": "option"}  # Focus on options; omit for all types

response = httpx.get(url, params=params, timeout=60)

print("Status Code:", response.status_code)

if response.status_code == 200:
    data = response.json()
    print("Available option roots/symbols:", data[:10])  # First 10 for brevity
else:
    print("Error:", response.text)