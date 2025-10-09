from dataclasses import dataclass

__all__ = ["Constants"]

@dataclass(frozen=True)
class Constants:
    PIPE_SEPARATOR: str = "|"