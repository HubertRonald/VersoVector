from __future__ import annotations

from dataclasses import dataclass

__all__ = ["Constants"]

@dataclass(frozen=True)
class Constants:
    UNDERLINE:str = '_'
    ENCODING:str = 'utf-8'
    EMPTY_STR:str = ''
    SPACE_STR:str = ' '
    COMMA_STR:str = ','
    PIPE_STR:str = '|'
    QUOTECHAR_STR:str = '"'
    ONE:int = 1
    ZERO:int = 0
    UNKNOWN: str = 'unknown'
    TAGS: str = 'tags'
    POET: str = 'poet'
    TITLE: str = 'title'
