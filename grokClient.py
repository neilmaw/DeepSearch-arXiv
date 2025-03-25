
from openai import OpenAI
class GrokClient:
    def __init__(self, q):
        self.client = OpenAI(
            api_key="[YOUR_API_KEY]",
            base_url="https://api.x.ai/v1",
        )
        self.q = q

    def request(self, context, query):
        prompt = f"read the following paper summaries:\n\n{str(context)}\n\nAnswer the question based on summaries: {query}"
        print(f"Generated prompt {prompt}")
        self.q.put({"step": "Generated prompt: "})
        self.q.put({"step": f"{prompt}"})

        completion = self.client.chat.completions.create(
            model="grok-2-latest",
            messages=[
                {"role": "user", "content": prompt},
            ],
        )
        result = completion.choices[0].message.content
        return result
