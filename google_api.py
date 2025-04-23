from google import genai
import os
from dotenv import load_dotenv

load_dotenv()

g_api = os.environ.get("GOOGLE_API_KEY")
client = genai.Client(api_key=g_api)

response = client.models.generate_content(
    model="gemini-2.5-pro-exp-03-25",
    contents="Explain how AI works in a few words",
)

print(response.text)