import requests

def perform_search(search_term):
    # 示例：使用本地API进行搜索
    url = 'http://localhost:5000/search'
    
    params = {
        'search_term': search_term,
        'page': 1,
        'per_page': 10
    }
    
    try:
        response = requests.get(url, params=params)
        if response.status_code == 200:
            data = response.json()
            print(f"Found {len(data['results'])} results.")
            return data
        else:
            raise ValueError(f"API request failed with status code {response.status_code}")
    except Exception as e:
        print(f"Error performing search: {e}")

# 示例使用
result = perform_search("搜索关键词")
if result:
    for item in result['results']:
        print(f"Item: {item.get('name', 'Unknown')} - Description: {item.get('description', 'Description Not Found')}")
else:
    print("No search results found.")
    