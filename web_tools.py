"""Инструменты поиска. Функции: tavily_post, search_web, extract_pages, search_wikipedia."""

import logging
import os

import httpx
from langchain_core.tools import tool
from pydantic import AnyHttpUrl, ValidationError


logger = logging.getLogger(__name__)


def tavily_post(endpoint: str, payload: dict) -> list[dict]:
    """Получает адрес раздела Tavily и данные для отправки. Возвращает список результатов."""
    response = httpx.post(
        f"https://api.tavily.com/{endpoint}",
        headers={"Authorization": f"Bearer {os.environ['TAVILY_API_KEY']}"},
        json=payload,
        timeout=25,
    )
    response.raise_for_status()
    return response.json()["results"]


@tool
def search_web(query: str) -> dict[str, str]:
    """Получает текст для поиска. Возвращает адреса страниц и их заголовки."""
    results = tavily_post("search", {"query": query, "max_results": 5})
    hits = {}
    for result in results:
        try:
            AnyHttpUrl(result["url"])
        except ValidationError:
            continue
        hits[result["url"]] = result["title"]
    return hits


@tool
def extract_pages(urls: list[str], query: str) -> dict[str, str]:
    """Получает адреса страниц и текст запроса. Возвращает отрывки по теме и их адреса."""
    results = tavily_post(
        "extract",
        {"urls": urls, "query": query, "chunks_per_source": 3, "format": "text"},
    )
    return {
        page["url"]: page["raw_content"][:8000]
        for page in results
        if page["url"] in urls and page.get("raw_content")
    }


@tool
def search_wikipedia(query: str) -> dict[str, tuple[str, str]]:
    """Принимает запрос. Возвращает адреса, заголовки и вводные фрагменты статей Википедии."""
    try:
        response = httpx.get(
            "https://ru.wikipedia.org/w/api.php",
            params={
                "action": "query",
                "generator": "search",
                "gsrsearch": query,
                "gsrnamespace": 0,
                "gsrlimit": 2,
                "prop": "extracts|info",
                "inprop": "url",
                "exintro": 1,
                "explaintext": 1,
                "format": "json",
                "formatversion": 2,
            },
            headers={"User-Agent": "FactChecker/1.0 (https://github.com/Viacheslav-hub-blip/factchecker)"},
            timeout=8,
        )
        response.raise_for_status()
        pages = response.json().get("query", {}).get("pages", [])
    except (httpx.HTTPError, ValueError) as error:
        logger.warning("Поиск в Википедии недоступен: %s", type(error).__name__)
        return {}
    return {
        page["fullurl"]: (page["title"], page["extract"][:2500])
        for page in pages
        if page.get("fullurl") and page.get("extract")
    }
