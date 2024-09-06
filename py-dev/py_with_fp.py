import httpx
from typing import Callable, Dict, Any, Optional, List, Tuple


def create_openai_payload(messages: List[Dict[str, str]]) -> Dict[str, Any]:
    """"""
    return {"model": "gpt-4o", "messages": messages, "temperature": 0.7}


def send_openai_request(payload: Dict[str, Any], api_key: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """"""
    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    try:
        return httpx.post(url=url, headers=headers, json=payload, timeout=None).raise_for_status().json(), None
    except httpx.HTTPStatusError as exc:
        return None, f"Error while requesting. Status code: {exc.response.status_code}, Message: {exc.response.text}"


def chat_with_openai(
    api_request_func: Callable[[Dict[str, Any], str], Tuple[Optional[Dict[str, Any]], Optional[str]]],
    create_payload_func: Callable[List[Dict[str, str]], Dict[str, Any]],
    messages: List[Dict[str, str]],
    api_key: str,
) -> Optional[Dict[str, Any]]:
    """"""
    payload = create_payload_func(messages)
    response, error_message = api_request_func(payload, api_key)

    if error_message:
        display_error_message(error_message)
        return None

    return response


def receive_user_message() -> str:
    return input("メッセージを入力してください (`/end` で終了): ")


def display_message(message: str) -> None:
    print(f"LLM: {message}")


def display_error_message(error_message: str) -> None:
    print(error_message)


def init_message() -> List[str]:
    return []


def update_message_history(messages: List[Dict[str, str]], new_message: Dict[str, str]) -> List[Dict[str, str]]:
    return messages + [new_message]


def interact_with_llm_in_chat(api_key: str) -> None:
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
