"""Приём сообщений и отправка ответов. Функции: lifespan, openapi_with_chat_example, chat, chat_error."""

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Lock
from uuid import UUID, uuid4

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from langchain_core.exceptions import OutputParserException
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from openai import APIError, APIStatusError, APITimeoutError
from langgraph.checkpoint.sqlite import SqliteSaver
from pydantic import BaseModel, ConfigDict, Field

from agent import Route, build_graph
from web_tools import extract_pages, search_web, search_wikipedia


logger = logging.getLogger("uvicorn.error")


class ChatRequest(BaseModel):
    """Сообщение пользователя и номер беседы, если она уже начата."""

    model_config = ConfigDict(str_strip_whitespace=True)
    message: str = Field(min_length=1, max_length=2000)
    thread_id: UUID | None = Field(
        default=None,
        description="Оставьте null для новой беседы. Для продолжения укажите номер из прошлого ответа.",
    )


class ChatResponse(BaseModel):
    """Ответ программы, номер беседы и выбранное действие."""

    thread_id: UUID
    route: Route
    answer: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Получает приложение, открывает SQLite и готовит его к проверке фактов."""
    load_dotenv()
    if not os.getenv("TAVILY_API_KEY"):
        raise RuntimeError("Укажите TAVILY_API_KEY в .env.")
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("Укажите OPENROUTER_API_KEY в .env.")
    model = ChatOpenAI(
        model=os.getenv("OPENROUTER_MODEL", "deepseek/deepseek-v4.1-flash-1017"),
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        timeout=45,
        extra_body={"provider": {"require_parameters": True}},
    )
    database = Path(os.getenv("FACTCHECKER_DB", str(Path(__file__).with_name("factchecker.sqlite3"))))
    if not database.is_absolute():
        database = Path(__file__).parent / database
    database.parent.mkdir(parents=True, exist_ok=True)
    with SqliteSaver.from_conn_string(str(database)) as checkpointer:
        checkpointer.setup()
        app.state.graph = build_graph(
            model, search_web, extract_pages, search_wikipedia, checkpointer
        )
        app.state.chat_lock = Lock()
        yield


app = FastAPI(title="FactChecker", lifespan=lifespan)
default_openapi = app.openapi


def openapi_with_chat_example() -> dict:
    """Возвращает описание запросов с примером новой беседы для Swagger."""
    schema = default_openapi()
    schema["components"]["schemas"]["ChatRequest"]["examples"] = [
        {"message": "Что ты умеешь?", "thread_id": None},
    ]
    return schema


app.openapi = openapi_with_chat_example


@app.post(
    "/chat",
    response_model=ChatResponse,
    responses={
        502: {"description": "Ошибка внешнего сервиса или неверный формат ответа"},
        503: {"description": "Внешний сервис ограничил частоту запросов"},
        504: {"description": "Внешний сервис не ответил вовремя"},
    },
)
def chat(body: ChatRequest, request: Request) -> ChatResponse:
    """Получает сообщение и данные запроса. Возвращает ответ и номер беседы."""
    thread_id = body.thread_id or uuid4()
    try:
        with request.app.state.chat_lock:
            result = request.app.state.graph.invoke(
                {"messages": [HumanMessage(content=body.message)]},
                config={"configurable": {"thread_id": str(thread_id)}},
            )
    except (APIError, httpx.HTTPError, OutputParserException) as error:
        raise chat_error(error) from error

    return ChatResponse(
        thread_id=thread_id,
        route=result["route"],
        answer=result["messages"][-1].text,
    )


def chat_error(error: Exception) -> HTTPException:
    """Получает ошибку проверки. Возвращает понятное сообщение для пользователя."""
    if isinstance(error, OutputParserException):
        logger.warning("OpenRouter: не удалось разобрать ответ модели.")
        return HTTPException(502, "OpenRouter вернул ответ в неожиданном формате. Повторите запрос.")

    service = "OpenRouter"
    status = error.status_code if isinstance(error, APIStatusError) else None
    if isinstance(error, httpx.HTTPError):
        if error.request.url.host == "api.tavily.com":
            service = "Tavily"
        if isinstance(error, httpx.HTTPStatusError):
            status = error.response.status_code

    logger.warning("%s: %s, код ответа: %s", service, type(error).__name__, status)
    if isinstance(error, (httpx.TimeoutException, APITimeoutError)):
        return HTTPException(504, f"{service} не ответил вовремя. Повторите запрос.")
    if status == 429:
        return HTTPException(503, f"{service} ограничил частоту запросов. Попробуйте чуть позже.")
    if status == 402:
        return HTTPException(502, f"{service} вернул код 402. Проверьте доступный лимит в личном кабинете.")
    if status in (401, 403):
        return HTTPException(502, f"{service} отклонил доступ ({status}). Проверьте ключ и права.")
    if status:
        return HTTPException(502, f"{service} вернул ошибку ({status}).")
    return HTTPException(502, f"Не удалось получить ответ от {service}.")
