import os
import logging

from fastapi import FastAPI, Request, Response

from openai import OpenAI

from slack_bolt import App
from slack_bolt.adapter.fastapi import SlackRequestHandler
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError


logging.basicConfig(level=logging.INFO)

app = App(
    token=os.getenv("YOUR_SLACK_BOT_TOKEN"),
    signing_secret=os.getenv("YOUR_SLACK_SIGNING_SECRET")
)
handler = SlackRequestHandler(app=app)
client = WebClient(token=os.getenv("YOUR_SLACK_BOT_TOKEN"))

@app.event("message")
def handle_message_events(body, say):
    '''Slack Appがメッセージを検知した際にリクエストが飛んでくるエンドポイント'''

    # Slack Appが受け取ったメッセージの内容を取り出す
    event = body['event']
    channel = event['channel']
    thread_ts = event.get('thread_ts', event['ts'])

    messages = get_thread_history(channel=channel, thread_ts=thread_ts)
    response = call_openai(messages=messages)

    # Slack にメッセージを返す
    say(text=response, channel=channel, thread_ts=thread_ts)

def get_thread_history(channel, thread_ts):
    '''メッセージが投稿されたスレッドの会話履歴を取得する'''
    try:
        result = client.conversations_replies(channel=channel, ts=thread_ts, inclusive=True, limit=50)
        messages = result['messages']
        return messages
    except SlackApiError as e:
        logging.error(f"Error fetching conversation history: {e.response['error']}")
        return []

# OpenAIのクライアント
openai = OpenAI(
    api_key=os.getenv("YOUR_OPENAI_API_KEY")
)

def call_openai(messages: list) -> str:
    '''OpenAIに送信するメッセージを組み立てる'''
    conversation = []
    for msg in messages:
        # OpenAIへのメッセージに含まれるロールについて、Slack Appからのメッセージは `assistant` とする
        role = "assistant" if msg['user'] == client.auth_test()['user_id'] else "user"
        conversation.append({"role": role, "content": msg['text']})

    completion = openai.chat.completions.create(
        model=os.getenv("CHAT_MODEL_GPT4o"),
        messages=[
            {
                "role": "system",
                "content": f"""
                    あなたは親切で丁寧かつ、有効的な対話型アシスタントです。
                    ユーザの質問や要求に対して、もし曖昧な点や複数の解釈が可能な場合は、ユーザの意図を明確にするための質問を返してください。
                    ユーザの意図を理解した上で、的確な応答を心がけてください。
                    """
            }
        ] + conversation
    )

    return completion.choices[0].message.content


api = FastAPI()

@api.post("/slack/events", name="slack events")
async def events(request: Request) -> Response:
    '''Slackからの疎通用リクエストを受け取るためのエンドポイント'''
    return await handler.handle(request)

@api.get("/slack/test")
async def test_endpoint():
    '''疎通確認用のエンドポイント'''
    return {"message": "Working!"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(api, host="0.0.0.0", port=3000)
