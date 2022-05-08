from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class Consumer(object):
    genre_preference: Dict[str, float] = None
    status: Optional[str] = None
    motivation: Optional[float] = None
    consume_type: Optional[str] = None
    can_view: int = 1
    richness: Optional[float] = None
    busyness: Optional[float] = None
    n_views: int = 0
    does_like_movie: bool = True
    children_genre: Optional[str] = None


@dataclass
class Movie(object):
    genre: Optional[str] = None
    target: Optional[str] = None
    promo_cost: Optional[int] = None
    broadcast_day: Optional[int] = None
