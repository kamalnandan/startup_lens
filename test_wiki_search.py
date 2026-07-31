import requests

def get_wikipedia_full(company_name):
    url = "https://en.wikipedia.org/w/api.php"
    headers = {
        "User-Agent": "StartupIntelligenceGraph/1.0 (kamal@example.com)"
    }
    params = {
        "action": "query",
        "titles": company_name,
        "prop": "extracts",
        "explaintext": True,
        "format": "json"
    }
    response = requests.get(url, headers=headers, params=params)
    pages = response.json()["query"]["pages"]
    page = next(iter(pages.values()))
    return page.get("extract", None)

content = get_wikipedia_full("Airbnb")
print(content[:3000])  # print first 3000 chars 
