"""FastAPI для FactChecker. Функции: lifespan, chat."""

import os
from contextlib import asynccontextmanager
from threading import Lock
from uuid import UUID, uuid4

import httpx
import httpx2
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from gigachat.exceptions import GigaChatException
from langchain_core.exceptions import OutputParserException
from langchain_core.messages import HumanMessage
from langchain_gigachat import GigaChat
from pydantic import BaseModel, ConfigDict, Field

from agent import Route, build_graph
from web_tools import create_web_tools


class ChatRequest(BaseModel):
    """Вход API — сообщение и необязательный идентификатор диалога."""

    model_config = ConfigDict(
        str_strip_whitespace=True,
        json_schema_extra={"examples": [{"message": "Что ты умеешь?"}]},
    )
    message: str = Field(min_length=1, max_length=2000)
    thread_id: UUID | None = Field(
        default=None, description="Для продолжения передайте thread_id из прошлого ответа."
    )


class ChatResponse(BaseModel):
    """Выход API — идентификатор диалога, выбранный маршрут и ответ агента."""

    thread_id: UUID
    route: Route
    answer: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Принимает приложение. Передаёт управление FastAPI с готовым графом и HTTP-клиентом."""
    load_dotenv()
    model = GigaChat(model=os.getenv("GIGACHAT_MODEL", "GigaChat-3-Pro"), timeout=45)
    with httpx.Client(
        base_url="https://api.tavily.com",
        headers={"Authorization": f"Bearer {os.environ['TAVILY_API_KEY']}"},
        timeout=25,
    ) as client:
        search_web, extract_pages = create_web_tools(client)
        app.state.graph = build_graph(model, search_web, extract_pages)
        app.state.chat_lock = Lock()
        yield


app = FastAPI(title="FactChecker", lifespan=lifespan)


@app.post("/chat", response_model=ChatResponse)
def chat(body: ChatRequest, request: Request) -> ChatResponse:
    """Принимает новое сообщение и HTTP-контекст. Возвращает ответ и UUID диалога."""
    thread_id = body.thread_id or uuid4()
    try:
        with request.app.state.chat_lock:
            result = request.app.state.graph.invoke(
                {"messages": [HumanMessage(content=body.message)]},
                config={"configurable": {"thread_id": str(thread_id)}},
            )
    except (GigaChatException, httpx.HTTPError, httpx2.HTTPError, OutputParserException) as error:
        raise HTTPException(status_code=502, detail="Ошибка GigaChat или Tavily.") from error

    return ChatResponse(
        thread_id=thread_id,
        route=result["route"],
        answer=result["messages"][-1].text,
    )
