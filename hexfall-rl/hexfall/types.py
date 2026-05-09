import random
from dataclasses import dataclass


@dataclass
class PlainBucket:
    color: str


@dataclass
class QuestionBucket:
    color: str
    revealed: bool


@dataclass
class Generator:
    facing: str      # 'up'|'down'|'left'|'right'
    remaining: int
    queue: list[str]  # queue[0] = next produced


@dataclass(frozen=True)
class Wall:
    pass


ReserveCell = PlainBucket | QuestionBucket | Generator | Wall  # Union; None = implicit empty


@dataclass
class BufferBucket:
    color: str
    capacity: int
    fill: int  # 0..capacity-1; bucket leaves at fill == capacity


@dataclass
class GameState:
    # Hex field
    field: dict[tuple[int, int], list[str]]   # (col, row) -> top-to-bottom slices

    # Buffer
    buffer_slots: int
    bucket_capacity: int
    buffer: list[BufferBucket | None]         # length = buffer_slots

    # Reserve
    reserve_rows: int
    reserve_cols: int
    reserve: list[list[ReserveCell | None]]   # reserve[row][col]

    # Meta
    color_set: frozenset[str]
    level_id: str

    # Runtime
    rng: random.Random
    quiescent: bool
