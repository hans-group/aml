from typing import TypeVar

T = TypeVar("T")
T_co = TypeVar("T_co", covariant=True)


def maybe_list(x: T | list[T]) -> list[T]:
    if isinstance(x, list):
        return x
    else:
        return [x]


def is_pkl(b: bytes) -> bool:
    first_two_bytes = b[:2]
    if first_two_bytes in (b"cc", b"\x80\x02", b"\x80\x03", b"\x80\x04", b"\x80\x05"):
        return True
    return False
