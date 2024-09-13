---
title: "関数型プログラミングで書いてみよう"
---


## 題材

関数型プログラミング実践の題材としては、比較的シンプルなOpenAI APIを使ってコンソール画面でチャットを行う、とってもシンプルなチャット機能の実装を採用します。
機能の詳細は下記です。

### 機能概要
- コンソール画面でユーザとOpenAIがチャットを行う
### 要件
- チャット開始から終了までは、チャット履歴が保存されること


# オブジェクト指向な書き方

まずはオブジェクト指向な書き方で、チャット機能を実装します。

`ChatBot`クラスを作成し、このクラス内にチャット機能を実装します。
`ChatBot`クラスの`start_chat`メソッドを呼び出すことで、OpenAI とのチャットを開始できるようにします。APIキーはクラスのフィールドに設定します。
`create_and_send_message`メソッドの引数にユーザメッセージを設定して、本メソッドを呼び出すことで、OpenAI へメッセージを送信します。

`ChatBot`クラスのイメージと利用方法は下記の通り。

```python
class ChatBot:
    def __init__(self, api_key: str):
        """クラスの初期化時にAPIキーを受け取る"""
        self.api_key = api_key

    def create_and_send_message(self, user_message: str) -> Optional[str]:
        """メッセージの組み立て、OpenAI API の呼び出し、チャット履歴の更新を行う"""

    def start_chat(self) -> None:
        """チャットを開始する"""
        while True:
          # ユーザが "/end" と入力するまで、チャット機能を継続する

if __name__ == "__main__":
    chatbot = ChatBot(api_key=os.getenv("OPENAI_API_KEY"))
    chatbot.start_chat()
```

:::message
外部APIへのリクエストをコード内でハンドリングすべく、OpenAI 提供の Python 用クライアントモジュールは利用せず、`httpx` を利用しています。
<!-- textlint-disable -->
:::
<!-- textlint-enable -->

<!-- textlint-disable -->
:::details コードの全体
<!-- textlint-enable -->
```python:oop.py
import httpx
from typing import Dict, Any, Optional, List


class ChatBot:
    def __init__(self, api_key: str):
        """
        ChatBotクラスのコンストラクタ。APIキーを設定し、メッセージ履歴を初期化します。

        Args:
            api_key (str): OpenAI APIキー。
        """
        self.api_key = api_key
        self.messages = []

    def create_and_send_message(self, user_message: str) -> Optional[str]:
        """
        ユーザーからのメッセージを基にOpenAIとやり取りし、アシスタントからのレスポンスを取得します。

        Args:
            user_message (str): ユーザーからのメッセージ。

        Returns:
            Optional[str]: アシスタントからのメッセージ。失敗した場合はNone。
        """
        # メッセージ履歴にユーザーのメッセージを追加
        self.messages.append({"role": "user", "content": user_message})

        # OpenAI APIに送信するペイロードを作成
        payload = {
            "model": "gpt-4o",
            "messages": self.messages,
            "temperature": 0.7,
        }

        # OpenAI APIへのリクエスト
        url = "https://api.openai.com/v1/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

        try:
            response = httpx.post(url=url, headers=headers, json=payload, timeout=None)
            response.raise_for_status()
            data = response.json()
        except httpx.HTTPStatusError as exc:
            self.display_message(f"エラーが発生しました。ステータスコード: {exc.response.status_code}, メッセージ: {exc.response.text}")
            return None

        # アシスタントのレスポンスを取得してメッセージ履歴に追加
        assistant_message = data["choices"][0]["message"]["content"]
        self.messages.append({"role": "assistant", "content": assistant_message})

        return assistant_message

    def display_message(self, message: str) -> None:
        """
        メッセージを表示します。エラーメッセージやアシスタントのメッセージの両方を処理します。

        Args:
            message (str): 表示するメッセージ。
        """
        print(message)

    def start_chat(self) -> None:
        """
        チャット形式でLLMと対話を行います。
        """
        while True:
            user_message = input("メッセージを入力してください (`/end` で終了): ")
            if user_message == "/end":
                break

            assistant_message = self.create_and_send_message(user_message)
            if assistant_message:
                self.display_message(f"LLM: {assistant_message}")


if __name__ == "__main__":
    import os
    from dotenv import load_dotenv

    load_dotenv()
    chatbot = ChatBot(api_key=os.getenv("OPENAI_API_KEY"))
    chatbot.start_chat()
```
:::



# 関数型プログラミングな書き方


## 本記事における関数型プログラミングとは？

本記事で関数型プログラミングを実践するにあたって、「[なっとく！関数型プログラミング](https://www.shoeisha.co.jp/book/detail/9784798179803)」から、下記の用語と定義を拝借します。

**関数型プログラミング**

> 関数型プログラミングとは イミュータブルな値を操作する純粋関数を利用するプログラミングである (P.70)

**純粋関数**

> - 戻り値は1つだけ
> - 引数のみに基づいて戻り値を計算する
> - 既存の値を変更しない
> 
> (P.46)





<!-- textlint-disable -->
:::details コードの全体
<!-- textlint-enable -->
```python:fp.py
import httpx
from typing import Callable, Dict, Any, Optional, List, Tuple


def create_openai_payload(messages: List[Dict[str, str]]) -> Dict[str, Any]:
    """
    OpenAI APIに送信するためのペイロードを作成します。

    Args:
        messages (List[Dict[str, str]]): 送信するメッセージのリスト。各メッセージは辞書形式で、'role'（ユーザーまたはアシスタント）と'content'（メッセージ内容）を含みます。

    Returns:
        Dict[str, Any]: OpenAI APIに送信するためのペイロードデータ。
    """
    return {"model": "gpt-4o", "messages": messages, "temperature": 0.7}


def send_openai_request(payload: Dict[str, Any], api_key: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    OpenAI APIにリクエストを送信し、レスポンスを返します。

    Args:
        payload (Dict[str, Any]): OpenAI APIに送信するペイロード。
        api_key (str): OpenAI APIキー。

    Returns:
        Tuple[Optional[Dict[str, Any]], Optional[str]]: 成功した場合はレスポンスのデータを含む辞書、エラーが発生した場合はエラーメッセージ。
    """
    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    try:
        return httpx.post(url=url, headers=headers, json=payload, timeout=None).raise_for_status().json(), None
    except httpx.HTTPStatusError as exc:
        return None, f"Error while requesting. Status code: {exc.response.status_code}, Message: {exc.response.text}"


def chat_with_openai(
    api_request_func: Callable[[Dict[str, Any], str], Tuple[Optional[Dict[str, Any]], Optional[str]]],
    create_payload_func: Callable[[List[Dict[str, str]]], Dict[str, Any]],
    messages: List[Dict[str, str]],
    api_key: str,
) -> Optional[Dict[str, Any]]:
    """
    OpenAIとチャットを行い、レスポンスを取得します。

    Args:
        api_request_func (Callable): APIリクエストを送信する関数。
        create_payload_func (Callable): ペイロードを作成する関数。
        messages (List[Dict[str, str]]): 送信するメッセージのリスト。
        api_key (str): OpenAI APIキー。

    Returns:
        Optional[Dict[str, Any]]: 成功した場合はレスポンスのデータ、失敗した場合はNone。
    """
    payload = create_payload_func(messages)
    response, error_message = api_request_func(payload, api_key)

    if error_message:
        display_error_message(error_message)
        return None

    return response


def receive_user_message() -> str:
    """
    ユーザーからの入力メッセージを取得します。

    Returns:
        str: ユーザーが入力したメッセージ。
    """
    return input("メッセージを入力してください (`/end` で終了): ")


def display_message(message: str) -> None:
    """
    アシスタントからのメッセージを表示します。

    Args:
        message (str): アシスタントからのメッセージ。
    """
    print(f"LLM: {message}")


def display_error_message(error_message: str) -> None:
    """
    エラーメッセージを表示します。

    Args:
        error_message (str): 表示するエラーメッセージ。
    """
    print(error_message)


def init_message() -> List[str]:
    """
    メッセージの履歴を初期化します。

    Returns:
        List[str]: 空のメッセージ履歴リスト。
    """
    return []


def update_message_history(messages: List[Dict[str, str]], new_message: Dict[str, str]) -> List[Dict[str, str]]:
    """
    メッセージ履歴に新しいメッセージを追加します。

    Args:
        messages (List[Dict[str, str]]): 現在のメッセージ履歴。
        new_message (Dict[str, str]): 追加する新しいメッセージ。

    Returns:
        List[Dict[str, str]]: 更新されたメッセージ履歴。
    """
    return messages + [new_message]


def interact_with_llm_in_chat(api_key: str) -> None:
    """
    チャット形式でLLMと対話を行います。

    Args:
        api_key (str): OpenAI APIキー。
    """
    messages = init_message()

    while True:
        user_message = receive_user_message()
        if user_message == "/end":
            break

        messages = update_message_history(messages=messages, new_message={"role": "user", "content": user_message})
        response = chat_with_openai(
            api_request_func=send_openai_request,
            create_payload_func=create_openai_payload,
            messages=messages,
            api_key=api_key,
        )

        if response is None:
            break

        if response:
            llm_message = response["choices"][0]["message"]["content"]
            messages = update_message_history(
                messages=messages, new_message={"role": "assistant", "content": llm_message}
            )

            display_message(llm_message)


if __name__ == "__main__":
    import os
    from dotenv import load_dotenv

    load_dotenv()
    interact_with_llm_in_chat(api_key=os.getenv("OPENAI_API_KEY"))
```
:::
