"""Локальный запуск FactChecker. Функция: main."""

import uvicorn


def main() -> None:
    """Запускает FastAPI на локальном порту 8000. Аргументов и результата нет."""
    uvicorn.run("app:app", host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
