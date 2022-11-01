import numpy as np

from functools import partial
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.metrics import Precision, Recall
from typing import List, Tuple

from .models import Consumer, Movie


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


def build_model(
    input_shape: Tuple[int, ...],
    output_shape: Tuple[int, ...],
    n_neurons: int = 524,
    name: str = None
):
    """学習モデルの構築
    """
    model = Sequential([
        Flatten(input_shape=input_shape, name='flatten'),
        Dense(n_neurons, activation='relu'),
        Dropout(0.2),
        Dense(output_shape, activation='softmax'),
    ], name=name)

    model.compile(
        optimizer='adam',
        # ラベルが2つ以上あるときに使用、ラベルは整数値で渡されることを想定
        loss='sparse_categorical_crossentropy',
        # ラベルは one-hot 表現
        # loss='categorical_crossentropy',
        metrics=[
            'acc',
            # 'accuracy',
            # Precision(),
            # Recall(),
        ],
    )

    return model


def params2labels(vec: np.ndarray, n_params: int, n_labels: int = 3):
    vec[:, :n_params] = numeric2label(vec[:, :n_params], n_labels)


def numeric2label(vec: np.ndarray, n_labels: int) -> np.ndarray:
    vec_min = vec.min(axis=0)
    vec_max = vec.max(axis=0)
    threshold = (vec_max - vec_min) / n_labels

    vec = vec - vec_min
    axis = 0 if vec.ndim == 1 else 1
    vec_labels = np.apply_along_axis(
        lambda arr: arr // threshold,
        axis=axis,
        arr=vec
    )

    on_bounds = vec_labels == n_labels
    vec_labels[on_bounds] = n_labels - 1

    return vec_labels
