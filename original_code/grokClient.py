
from openai import OpenAI
class GrokClient:
    def __init__(self):
        self.client = OpenAI(
            api_key="xai-r9fWsSSsbRMo3tv7ZHZmF2BRLhdAoiNWHGmm5dTmwL1oUBJp0v719wkSyiJnieiBAXycUOSEt6buUSgd",
            base_url="https://api.x.ai/v1",
        )

    def request(self, context, query):
        prompt = f"read the following paper summaries:\n\n{str(context)}\n\nAnswer the question based on summaries: {query}"
        print(f"Generated prompt {prompt}")
        completion = self.client.chat.completions.create(
            model="grok-2-latest",
            messages=[
                {"role": "user", "content": prompt},
            ],
        )
        result = completion.choices[0].message.content
        return result
