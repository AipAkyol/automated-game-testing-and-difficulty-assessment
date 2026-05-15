import random
from dataclasses import dataclass, field


@dataclass
class PlainBucket:
    color: str


@dataclass
class QuestionBucket:
    color: str
    revealed: bool


@dataclass
class IceBucket:
    row: int
    col: int
    color: str          # hidden from agent until thawed
    thaw_threshold: int # iceCapacity from JSON
    thawed: bool = False


@dataclass
class Generator:
    facing: str      # 'up'|'down'|'left'|'right'
    remaining: int
    queue: list[str]  # queue[0] = next produced


@dataclass(frozen=True)
class Wall:
    pass


@dataclass
class Pin:
    origin_row: int
    origin_col: int
    direction: str      # "Up", "Down", "Left", "Right"
    block_count: int    # 0 means extend to grid edge
    destroyed: bool = False


ReserveCell = PlainBucket | QuestionBucket | IceBucket | Generator | Wall  # None = implicit empty


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

    # Pins overlay the reserve grid; stored separately from cell content.
    pins: list[Pin] = field(default_factory=list)

    # Move counter starts at 0 and increments on each player action; gates ice thaw.
    move_counter: int = 0
