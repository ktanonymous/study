"""2層のネットワークを利用して,
ダミーデータを教師データとした映画鑑賞に関する行動予測を行う.
"""
import numpy as np
import os
import pandas as pd

from collections import defaultdict
from create_data_no_plot import main as create_data
from models import Consumer, Movie
from network import Network
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from typing import List


def main(period: int = 100):
    dir_name = os.path.dirname(__file__)
    input_file = os.path.normpath(os.path.join(dir_name, '../json/spec.json'))
    view_data, consumers, movies = create_data(input_file, period)
    inputs_consumer = create_table(view_data, consumers, movies, period)

    nn = Sequential([
        Flatten(input_shape=inputs_consumer[0].shape, name='flatten_layer'),
        Dense(256, activation='relu'),
        Dropout(0.2),
        Dense(len(movies), activation='softmax'),
    ], name='test_model')
    nn.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )
    print(inputs_consumer.shape)
    print(np.apply_along_axis(np.sum, 0, view_data[:, 0, :]).shape)
    history = nn.fit(
        inputs_consumer,
        np.apply_along_axis(
            func1d=np.sum,
            axis=0,
            arr=view_data[:, 0, :],
        ),
        batch_size=128, epochs=20, validation_split=0.2, verbose=1,
    )
    print(history.history.keys())
    for i, data in enumerate(inputs_consumer):
        acc = history.history['accuracy'][i]
        movie_idx = None if all(data == 0) else np.argmax(data)
        genre = movies[movie_idx].genre if movie_idx else None
        preference = consumers[0].genre_preference[genre] if genre else None
        print(genre, preference, acc)


def create_table(
    view_data: np.ndarray,
    consumers: List[Consumer],
    movies: List[Movie],
    period: int,
) -> np.ndarray:
    # 各映画の公開ラベル(size=(period, n_movies))を付与
    INTERVAL_MOVIE = 70
    is_showings = np.zeros((period, len(movies)))
    for i, movie in enumerate(movies):
        broadcast_day = movie.broadcast_day
        final_day = broadcast_day + INTERVAL_MOVIE
        is_showings[broadcast_day:final_day, i] = 1
    # 鑑賞情報を作品 ID に変換する(size=(n_consumers, period), range=(-1, n_movies))
    view_ids = np.apply_along_axis(
        func1d=lambda vec: -1 if all(vec == 0) else np.argmax(vec),
        axis=2,
        arr=view_data.transpose(1, 2, 0),
    )

    # 鑑賞状況を踏まえた公開情報ラベルをマスク化する
    # NOTE: 現在は特定の消費者のみに注目
    consumer = consumers[0]
    # 公開情報ラベルに消費者の鑑賞ラベルを付加する(size=(period, n_movies+1))
    is_showings_with_viewed_id = np.concatenate(
        [is_showings, view_ids[consumer.id_num].reshape(-1, 1)],
        axis=1,
    )
    # 公開状況の組のマスク(size=(label-kinds, n_movies))
    mask_showing = np.unique(is_showings_with_viewed_id, axis=0)[:, :-1]

    # 各映画に該当するジャンル選好度をマスクする(size=(label-kinds, n_movies))
    genre_preference = np.array([
        consumer.genre_preference[movie.genre]
        for movie in movies
    ]).reshape(1, -1)
    genre_preferences_masked = np.multiply(mask_showing, genre_preference)

    # 個人属性ネットワークへの入力を作成する(size=(label-kinds, n_movies))
    inputs = np.apply_along_axis(
        func1d=lambda vec:
            np.concatenate([
                vec,
                # consumer.consumer_type,
                np.full(vec.shape, consumer.richness),
                # np.full(vec.shape, consumer.n_views),
                # np.full(vec.shape, consumer.does_like_movie),
                # np.full(vec.shape, consumer.childre_genre),
            ]),
        axis=0,
        arr=genre_preferences_masked,
    )
    return inputs


if __name__ == '__main__':
    main()
