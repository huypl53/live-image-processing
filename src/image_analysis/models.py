import numpy as np
from typing import Any, Dict, List, Literal, Optional, Tuple, TypedDict

class Box(TypedDict):
    x: int
    y: int
    width: int
    height: int

class BoxComponent(TypedDict):
    id: int
    type: str
    bbox: Box
    area: int

class SegResult(TypedDict):
    steps: Dict[str, np.ndarray]
    components: List[BoxComponent]
    total_components: int