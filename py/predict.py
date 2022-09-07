"""2層のネットワークを利用して,
ダミーデータを教師データとした映画鑑賞に関する行動予測を行う.
"""
import numpy as np
import os
import time

from collections import defaultdict
from functools import partial
from models import Consumer, Movie
from network import Network
from statistics import mean
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from typing import List, Generator

from create_data_no_plot import main as create_data


def main(period: int = 100):
    dir_name = os.path.dirname(__file__)
    input_file = os.path.normpath(os.path.join(dir_name, '../json/spec.json'))
    # データ生成(period=100, n_consumers=800): 約 20 ~ 30 sec
    start = time.time()
    view_data, consumers, movies = create_data(input_file, period)
    n_consumers = len(consumers)
    print(
        f"Time to create data(period: {period}, consumers: {n_consumers}) -> "
        f"{time.time() - start:.1f} sec"
    )
    # テーブル作成(period=100, n_consumers=800): 約 0.5 sec
    start = time.time()
    inputs_consumer, is_showings = create_table(
        view_data, consumers, movies, period
    )
    print(
        f"Time to create table(period: {period}, consumers: {n_consumers}) -> "
        f"{time.time() - start:.1f} sec"
    )
    n_movies = len(movies)
    n_labels = n_movies + 1

    # モデルの定義
    nn = Sequential([
        # 鑑賞作品 ID を入力に入れると予測の際に入力のサイズが足りなくなる
        Flatten(input_shape=inputs_consumer[0, :-1].shape, name='flatten'),
        Dense(128, activation='relu'),
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
    n_params = inputs_consumer[:, :-(n_movies+1)].shape[-1]
    print(f"accuracy: {history.history['accuracy'][-1]}")
    print('\ncheck_accuracy')
    # モデルチェック(period=100, n_consumers=800): 約 5+ min
    start = time.time()
    check_accuracy(
        nn, movies, consumers, view_data, is_showings,
        n_params=n_params
    )
    print(
        f"Time to check accuracy(period: {period}, consumers: {n_consumers})"
        f" -> {time.time() - start:.1f} sec"
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

    # ダミーデータを使って実際に予測してみる
    # NOTE: 現状は全てで学習した後に全てをチェック対象にしている("テスト"より"チェック")
    is_correct_predicts = defaultdict(list)
    n_misses = 0
    predict_ids = []
    print('consumer_ID: ', end='')
    for consumer, view_ids in zip(consumers, view_ids_all):
        if consumer.id_num % 80 == 0:
            print(consumer.id_num, end=', ', flush=True)
        view_ids_one_hot = np.eye(period)[view_ids][:, :n_movies+1]

        # 公開情報を選好度で上書きし、鑑賞作品 ID を付与する(重複は削除)
        preference = np.array([
            consumer.genre_preference[movie.genre]
            for movie in movies
        ]).reshape(1, -1)
        preferences_showing = np.multiply(
            is_showings,
            preference
        )
        preferences_showing_with_viewed_id = np.unique(
            np.concatenate(
                [preferences_showing, view_ids.reshape(-1, 1)],
                axis=1,
            ),
            axis=0
        )

        # 個人パラメータを追加して日毎に予測及び正解のチェックを行う
        params = get_params(consumer)
        n_datumn, _ = preferences_showing_with_viewed_id.shape
        datumn = np.concatenate([
            (np.ones((n_datumn, n_params)) * params).reshape(-1, n_params),
            preferences_showing_with_viewed_id,
        ], axis=1)
        iter_viewing = zip(view_ids, view_ids_one_hot, datumn)
        # print('mispredicted:')
        for view_id, view_correct, data in iter_viewing:
            view_predict = model.predict(data[:-1].reshape(1, -1))
            id_predict = np.argmax(view_predict)
            predict_ids.append(id_predict)
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
            #     print(
            #         f"correct-ID: {view_id}({view_genre}), "
            #         f"predict-ID: {id_predict}({genre_predict})"
            #     )

    print(f"\nmispredicted: {n_misses}\n")
    print('genre: n_histories, n_corrects')
    for genre, history in is_correct_predicts.items():
        print(f"{genre}: {len(history)}, {sum(history)}")
    print(f"If network predicted all as same: {len(set(predict_ids)) == 1}")


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
    view_ids_all = np.apply_along_axis(
        func1d=lambda vec: n_movies if (vec == 0).all() else np.argmax(vec),
        axis=2,
        arr=view_data.transpose(1, 2, 0),
    )

    _create_table_fixed = partial(
        _create_table,
        movies=movies,
        is_showings=is_showings,
    )
    inputs = np.concatenate([
        inputs_consumer for inputs_consumer
        in _create_table_fixed(consumers=consumers, view_ids_all=view_ids_all)
    ], axis=0)

    return inputs, is_showings


def _create_table(
    consumers: List[Consumer],
    view_ids_all: np.ndarray,
    movies: List[Movie],
    is_showings: np.ndarray,
) -> Generator[np.ndarray, None, None]:
    # 鑑賞状況を踏まえた公開情報ラベルをマスク化する
    # 公開情報ラベルに消費者の鑑賞ラベルを付加する(size=(period, n_movies+1))
    for consumer, view_ids in zip(consumers, view_ids_all):
        is_showings_with_viewed_id = np.concatenate(
            [is_showings, view_ids.reshape(-1, 1)],
            axis=1,
        )
        # 公開状況の組のマスク(size=(label-kinds, n_movies+1))
        # NOTE: 映画を見たかどうかも含める
        mask_showing_with_viewed_id = np.unique(
            is_showings_with_viewed_id, axis=0
        )
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
        inputs_consumer = np.concatenate([
            np.ones((n_label_kinds, n_params)) * params,
            genre_preferences_masked_with_viewed_id,
        ], axis=1)

        yield inputs_consumer


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
