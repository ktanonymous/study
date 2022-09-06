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
from statistics import mean
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from typing import List


def main(period: int = 400):
    dir_name = os.path.dirname(__file__)
    input_file = os.path.normpath(os.path.join(dir_name, '../json/spec.json'))
    view_data, consumers, movies = create_data(input_file, period)
    inputs_consumer, is_showings = create_table(
        view_data, consumers, movies, period
    )
    n_movies = len(movies)
    n_labels = n_movies + 1

    # モデルの定義
    nn = Sequential([
        # 鑑賞作品 ID を入力に入れると予測の際に入力のサイズが足りなくなる
        Flatten(input_shape=inputs_consumer[0, :-1].shape, name='flatten'),
        Dense(512, activation='relu'),
        Dropout(0.2),
        # one-hot に合わせて「何も見ない」次元を出力に追加する
        Dense(n_labels, activation='softmax'),
    ], name='test_model')
    nn.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )

    # モデルの学習
    learning_data = inputs_consumer[:, :-1]
    # 「何も見ない」次元を追加したラベル(range=(0, n_movies))
    teaching_data = inputs_consumer[:, -1].astype(int)
    history = nn.fit(
        learning_data,
        teaching_data,
        batch_size=128, epochs=20, validation_split=0.2, verbose=1,
    )

    # モデルの詳細及び精度の検証
    nn.summary()
    print(f"accuracy: {history.history['accuracy'][-1]}")
    print('\ncheck_accuracy')
    check_accuracy(
        nn, movies, consumers, view_data, is_showings,
        n_params=inputs_consumer[:, :-(n_movies+1)].shape[-1]
    )
    print(f"\nteaching-data:\n{inputs_consumer[:, -1].astype(int)}")


def check_accuracy(model, movies: List[Movie], consumers: List[Consumer],
                   view_data: np.ndarray, is_showings: np.ndarray,
                   n_params: int):
    # パラメータ情報を取得
    n_movies, n_consumers, period = view_data.shape

    # 「何も見ない」次元を追加した鑑賞作品 ID (size=(n_consumer, period))
    view_ids_all = np.apply_along_axis(
        func1d=lambda vec: n_movies if (vec == 0).all() else np.argmax(vec),
        axis=2,
        arr=view_data.transpose(1, 2, 0),
    )

    # NOTE: 現在は特定の消費者のみに注目
    consumer = consumers[10]
    view_ids = view_ids_all[consumer.id_num]
    view_ids_one_hot = np.eye(period)[view_ids][:, :n_movies+1]

    # 公開情報を選好度で上書きし、鑑賞作品 ID を付与する
    preference = np.array([
        consumer.genre_preference[movie.genre]
        for movie in movies
    ]).reshape(1, -1)
    preferences_showing = np.multiply(
        is_showings,
        preference
    )
    preferences_showing_with_viewed_id = np.concatenate(
        [preferences_showing, view_ids.reshape(-1, 1)],
        axis=1,
    )

    # 個人パラメータを追加して日毎に予測及び正解のチェックを行う
    is_correct_predicts = defaultdict(list)
    params = get_params(consumer)
    datumn = np.concatenate([
        (np.ones((period, n_params)) * params).reshape(-1, n_params),
        preferences_showing_with_viewed_id,
    ], axis=1)
    iter_viewing = zip(view_ids, view_ids_one_hot, datumn)
    n_misses = 0
    print('mispredicted:')
    for view_id, view_correct, data in iter_viewing:
        view_predict = model.predict(data[:-1].reshape(1, -1))
        id_predict = np.argmax(view_predict)
        is_correct_predict = view_id == id_predict
        if view_id == n_movies:
            view_genre = 'Nothing'
        else:
            view_genre = movies[view_id].genre
        if id_predict == n_movies:
            genre_predict = 'Nothing'
        else:
            genre_predict = movies[id_predict].genre
        is_correct_predicts[view_genre].append(int(is_correct_predict))

        if not is_correct_predict:
            n_misses += 1
            print(
                f"correct-ID: {view_id}({view_genre}), "
                f"predict-ID: {id_predict}({genre_predict})"
            )

    print(f"mispredicted: {n_misses}\n")
    print('genre: n_histories, n_corrects')
    for genre, history in is_correct_predicts.items():
        print(f"{genre}: {len(history)}, {sum(history)}")


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
    # 鑑賞情報を作品 ID に変換する(size=(n_consumers, period), range=(0, n_movies+1))
    n_movies = len(movies)
    view_ids = np.apply_along_axis(
        func1d=lambda vec: n_movies if (vec == 0).all() else np.argmax(vec),
        axis=2,
        arr=view_data.transpose(1, 2, 0),
    )

    # 鑑賞状況を踏まえた公開情報ラベルをマスク化する
    # NOTE: 現在は特定の消費者のみに注目
    consumer = consumers[10]
    # 公開情報ラベルに消費者の鑑賞ラベルを付加する(size=(period, n_movies+1))
    is_showings_with_viewed_id = np.concatenate(
        [is_showings, view_ids[consumer.id_num].reshape(-1, 1)],
        axis=1,
    )
    # 公開状況の組のマスク(size=(label-kinds, n_movies+1))
    mask_showing_with_viewed_id = np.unique(is_showings_with_viewed_id, axis=0)
    n_label_kinds = len(mask_showing_with_viewed_id)

    # 各映画に該当するジャンル選好度をマスクする(size=(label-kinds, n_movies))
    genre_preference = np.array([
        consumer.genre_preference[movie.genre]
        for movie in movies
    ]).reshape(1, -1)
    genre_preferences_masked = np.multiply(
        mask_showing_with_viewed_id[:, :-1],
        genre_preference,
    )
    genre_preferences_masked_with_viewed_id = np.concatenate(
        [
            genre_preferences_masked,
            mask_showing_with_viewed_id[:, -1].reshape(-1, 1)
        ],
        axis=1,
    )

    # 個人属性ネットワークへの入力を作成する(size=(label-kinds, n_parameters+n_movies+1))
    # 鑑賞情報も含める
    params = get_params(consumer)
    n_params = len(params)
    inputs = np.concatenate([
        np.ones((n_label_kinds, n_params)) * params,
        genre_preferences_masked_with_viewed_id,
    ], axis=1)
    return inputs, is_showings


def get_params(consumer: Consumer) -> np.ndarray:
    params = np.array([
        # consumer.consumer_type,
        consumer.richness,
        # consumer.n_views,
        # consumer.does_like_movie,
        # consumer.childre_genre,
    ])

    return params


if __name__ == '__main__':
    main()
