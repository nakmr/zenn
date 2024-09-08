import httpx
from pydantic import BaseModel, conlist, StringConstraints, Field
from typing import Callable, Dict, Any, Optional, List, Tuple, Union, Annotated


class Message(BaseModel):
    """
    OpenAI APIに送信されるメッセージを表すモデル。

    Attributes:
        role (str): 'user' または 'assistant' という役割を持つ文字列。
        content (str): メッセージの内容。
    """

    role: Annotated[str, StringConstraints(pattern=r"^(user|assistant)$")]
    content: str


class OpenAIPayload(BaseModel):
    """
    OpenAI APIにリクエストする際のペイロードを表すモデル。

    Attributes:
        model (str): モデル名 (例: gpt-4o)。
        messages (List[Message]): 少なくとも1つ以上のメッセージリスト。
        temperature (float): モデルの創造性に影響を与える0.0から1.0の範囲の値。
    """

    model: str
    messages: conlist(Message, min_length=1)  # 少なくとも1つ以上のMessageが必要
    temperature: Annotated[float, Field(ge=0.0, le=1.0)]


class Choice(BaseModel):
    """
    OpenAI APIレスポンスに含まれる選択肢を表すモデル。

    Attributes:
        message (Message): 選択されたメッセージ。
    """

    message: Message


class OpenAIResponse(BaseModel):
    """
    OpenAI APIからのレスポンスを表すモデル。

    Attributes:
        choices (List[Choice]): モデルが返すメッセージのリスト。
    """

    choices: List[Choice]


class APIError(BaseModel):
    """
    APIリクエスト時に発生するエラーを表すモデル。

    Attributes:
        status_code (int): HTTPステータスコード。
        message (str): エラーメッセージ。
    """

    status_code: int
    message: str


ErrorMessage = Union[APIError, str]


def create_openai_payload(messages: List[Message]) -> Dict[str, Any]:
    """
    OpenAI APIに送信するペイロードを作成します。

    Args:
        messages (List[Message]): チャットメッセージのリスト。

    Returns:
        Dict[str, Any]: OpenAI APIに渡すペイロードを含む辞書形式のデータ。
    """

    # pydanticモデルは辞書形式に変換してからJSONとして送信する必要がある。
    # 具体的には、messagesリストの各Messageオブジェクトをmessage.dict()に変換する。
    return {"model": "gpt-4o", "messages": [message.dict() for message in messages], "temperature": 0.7}


def send_openai_request(
    payload: OpenAIPayload, api_key: str
) -> Tuple[Optional[OpenAIResponse], Optional[ErrorMessage]]:
    """
    OpenAI APIにリクエストを送信します。

    Args:
        payload (OpenAIPayload): OpenAI APIに送信するペイロード。
        api_key (str): OpenAI APIキー。

    Returns:
        Tuple[Optional[OpenAIResponse], Optional[ErrorMessage]]:
            - 成功時にはOpenAI APIからのレスポンスを返します。
            - エラー時にはエラーメッセージを返します。
    """

    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    try:
        response = httpx.post(url=url, headers=headers, json=payload, timeout=None)
        response.raise_for_status()
        # send_openai_request関数内でAPIレスポンスを辞書形式で受け取っているため、OpenAIResponseとして処理する際に辞書がそのまま渡される。
        # httpxが返すレスポンスを直接json()でパースしているため、pydanticモデルにマッピングする必要がある。
        # そのため、APIからのレスポンスをOpenAIResponseモデルに変換する。
        return OpenAIResponse(**response.json()), None
    except httpx.HTTPStatusError as exc:
        return None, APIError(status_code=exc.response.status_code, message=exc.response.text)


def chat_with_openai(
    api_request_func: Callable[[OpenAIPayload, str], Tuple[Optional[OpenAIResponse], Optional[str]]],
    create_payload_func: Callable[[List[Message]], OpenAIPayload],
    messages: List[Message],
    api_key: str,
) -> Optional[OpenAIResponse]:
    """
    OpenAI APIとやり取りを行い、チャットのレスポンスを取得します。

    Args:
        api_request_func (Callable[[OpenAIPayload, str], Tuple[Optional[OpenAIResponse], Optional[str]]]):
            APIリクエストを送信する関数。
        create_payload_func (Callable[[List[Message]], OpenAIPayload]):
            ペイロードを作成する関数。
        messages (List[Message]): チャットメッセージのリスト。
        api_key (str): OpenAI APIキー。

    Returns:
        Optional[OpenAIResponse]: OpenAIからのレスポンス。エラー時にはNoneを返します。
    """

    payload = create_payload_func(messages)
    response, error_message = api_request_func(payload, api_key)

    if error_message:
        display_error_message(error_message)
        return None

    return response


def receive_user_message() -> str:
    """
    ユーザーからのメッセージを受け取ります。

    Returns:
        str: ユーザーが入力したメッセージ。
    """

    return input("メッセージを入力してください (`/end` で終了): ")


def display_message(message: str) -> None:
    """
    メッセージを表示します。

    Args:
        message (str): 表示するメッセージ。
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
    チャットメッセージの履歴を初期化します。

    Returns:
        List[str]: 空のメッセージリスト。
    """

    return []


def update_message_history(messages: List[Message], new_message: Message) -> List[Message]:
    """
    メッセージ履歴を更新し、新しいメッセージを追加します。

    Args:
        messages (List[Message]): 現在のメッセージ履歴。
        new_message (Message): 追加する新しいメッセージ。

    Returns:
        List[Message]: 更新されたメッセージ履歴。
    """

    return messages + [new_message]


def interact_with_llm_in_chat(api_key: str) -> None:
    """
    OpenAIとチャット形式で対話を行います。

    Args:
        api_key (str): OpenAI APIキー。
    """

    messages = init_message()

    while True:
        user_message = receive_user_message()
        if user_message == "/end":
            break

        messages = update_message_history(messages=messages, new_message=Message(role="user", content=user_message))
        response = chat_with_openai(
            api_request_func=send_openai_request,
            create_payload_func=create_openai_payload,
            messages=messages,
            api_key=api_key,
        )

        if response is None:
            break

        if response:
            llm_message = response.choices[0].message.content
            messages = update_message_history(
                messages=messages, new_message=Message(role="assistant", content=llm_message)
            )

            display_message(llm_message)


if __name__ == "__main__":
    import os
    from dotenv import load_dotenv

    load_dotenv()
    interact_with_llm_in_chat(api_key=os.getenv("OPENAI_API_KEY"))
