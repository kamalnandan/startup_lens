import requests
from app_config import get_required_setting

api_key = get_required_setting("AZURE_OPENAI_API_KEY")
endpoint = get_required_setting("AZURE_OPENAI_ENDPOINT").rstrip("/")
deployment = get_required_setting("AZURE_OPENAI_DEPLOYMENT")
api_version = get_required_setting("AZURE_OPENAI_API_VERSION")
print("API key loaded: YES")

# Test API call
url = f"{endpoint}/openai/deployments/{deployment}/chat/completions?api-version={api_version}"
response = requests.post(
    url,
    headers={
        "Content-Type": "application/json",
        "api-key": api_key
    },
    json={
        "messages": [{"role": "user", "content": "Say hello"}],
        "max_tokens": 10
    },
    timeout=30
)
print(f"Status: {response.status_code}")
print(f"Response: {response.json()}")