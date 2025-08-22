import re

def parse_query(query: str):
    return {
        "age": int(re.search(r'(\d+)[ -]?year', query).group(1)) if re.search(r'(\d+)[ -]?year', query) else None,
        "procedure": re.search(r'(\w+\s+surgery)', query).group(1) if re.search(r'(\w+\s+surgery)', query) else None,
        "location": re.search(r'in\s([A-Za-z]+)', query).group(1) if re.search(r'in\s([A-Za-z]+)', query) else None,
        "policy_duration": int(re.search(r'(\d+)[ -]?month', query).group(1)) if re.search(r'(\d+)[ -]?month', query) else None,
        "gender": "male" if "male" in query.lower() else "female"
    }
