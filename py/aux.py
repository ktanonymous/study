import numpy as np

from functools import partial
from typing import List

from models import Consumer, Movie


def get_preferences_all(
    consumers: List[Consumer],
    movies: List[Movie]
) -> np.ndarray:
    get_preferences_fixed = partial(
        get_preferences,
        consumers=consumers
    )
    preferences_all = np.apply_along_axis(
        func1d=lambda movie: get_preferences_fixed(movie=movie[0]),
        axis=0,
        arr=np.array(movies).reshape(1, -1)
    )

    return preferences_all


def get_preferences(consumers: List[Consumer], movie: Movie) -> np.ndarray:
    _get_preferences_fixed = partial(
        get_preference,
        movie=movie
    )
    preferences = np.apply_along_axis(
        func1d=lambda consumer: _get_preferences_fixed(consumer=consumer[0]),
        axis=0,
        arr=np.array(consumers).reshape(1, -1)
    )

    return preferences


def get_preference(consumer: Consumer, movie: Movie) -> float:
    return consumer.genre_preference[movie.genre]
