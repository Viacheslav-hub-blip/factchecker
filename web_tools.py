"""Инструменты Tavily. Функции: create_web_tools, search_web, extract_pages."""

import httpx
from langchain_core.tools import BaseTool, tool
from pydantic import AnyHttpUrl, ValidationError


def create_web_tools(client: httpx.Client) -> tuple[BaseTool, BaseTool]:
    """Принимает настроенный HTTP-клиент. Возвращает инструменты поиска и чтения страниц."""

    @tool
    def search_web(query: str) -> dict[str, str]:
        """Принимает поисковый запрос. Возвращает словарь адресов и заголовков страниц."""
        response = client.post("/search", json={"query": query, "max_results": 5})
        response.raise_for_status()
        hits = {}
        for result in response.json()["results"]:
            try:
                AnyHttpUrl(result["url"])
            except ValidationError:
                continue
            hits[result["url"]] = result["title"]
        return hits

    @tool
    def extract_pages(urls: list[str], query: str) -> dict[str, str]:
        """Принимает адреса и запрос. Возвращает адреса и релевантные фрагменты страниц."""
        response = client.post(
            "/extract",
            json={"urls": urls, "query": query, "chunks_per_source": 3, "format": "text"},
        )
        response.raise_for_status()
        return {
            page["url"]: page["raw_content"][:8000]
            for page in response.json()["results"]
            if page["url"] in urls and page.get("raw_content")
        }

    return search_web, extract_pages
