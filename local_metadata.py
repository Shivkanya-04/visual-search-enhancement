import requests
import json

session = requests.Session()
HEADERS = {"Content-Type": "application/json"}

def call_local_llm(prompt):
    url = "http://127.0.0.1:1234/v1/chat/completions"

    payload = {
        "model": "phi-3.5-mini-instruct",
        "temperature": 0,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "product_metadata",
                "schema": {
                    "type": "object",
                    "properties": {
                        "title": { "type": "string" },
                        "bullet_points": {
                            "type": "array",
                            "items": { "type": "string" }
                        },
                        "description": { "type": "string" },
                        "style_summary": { "type": "string" },
                        "seo_tags": {
                            "type": "array",
                            "items": { "type": "string" }
                        }
                    },
                    "required": [
                        "title",
                        "bullet_points",
                        "description",
                        "style_summary",
                        "seo_tags"
                    ]
                }
            }
        },
        "messages": [
            {
                "role": "system",
                "content": "You only output JSON that matches the schema. No explanation or extra text."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    }

    response = session.post(url, headers=HEADERS, json=payload)
    data = response.json()

    if "choices" not in data:
        print("RAW:", data)
        raise ValueError("No choices returned from LLM")

    return data["choices"][0]["message"]["content"]


def generate_local_metadata(attributes):
    prompt = f"""
Generate product metadata for this women's top:

{json.dumps(attributes, indent=2)}

Rules:
- Description must be 1–2 concise sentences only.
- Style summary must be exactly 1 sentence.
- Bullet points must be short phrases (max 6 words).
- Title must be under 12 words.
- SEO tags must be 3–5 short tags.
- Do NOT add information not implied by the attributes.
"""
    text = call_local_llm(prompt)
    return json.loads(text)
