"""2層のネットワークを利用して,
ダミーデータを教師データとした映画鑑賞に関する行動予測を行う.
"""
import numpy as np
import os
import tensorflow as tf
import time

from collections import defaultdict
from functools import partial
from statistics import mean
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.math import confusion_matrix
from tensorflow.keras.metrics import Precision, Recall
from typing import List, Generator

from aux import get_preferences_all
from create_data_no_plot import main as create_data
from models import Consumer, Movie
from network import Network


def main(
    period: int = 100,
    use_csv: bool = False
) -> Tuple[np.ndarray, np.ndarray, List[int], tf.Tensor]:
    dir_name = os.path.dirname(__file__)
    input_file = os.path.normpath(os.path.join(dir_name, '../json/spec.json'))
    # データ生成(period=100, n_consumers=800): 約 5 ~ 8 sec
    # データ生成(period=400, n_consumers=800): 約 14 ~ 20 sec
    start = time.time()
    view_data, consumers, movies = create_data(input_file, period, use_csv)
    elapsed_time = time.time() - start
    n_consumers = len(consumers)
    genre_preferences_all = get_preferences_all(consumers, movies)
    print(
        f"Time to create data(period: {period}, consumers: {n_consumers}) -> "
        f"{elapsed_time:.1f} sec"
    )
    # テーブル作成(period=100, n_consumers=800): 約 0.5 sec
    # テーブル作成(period=400, n_consumers=800): 約 2 sec
    n_movies = len(movies)
    # 「何も見ない」次元を追加した鑑賞作品 ID (size=(n_consumers, period))
    view_ids_all = np.apply_along_axis(
        func1d=lambda vec: n_movies if (vec == 0).all() else np.argmax(vec),
        axis=2,
        arr=view_data.transpose(1, 2, 0),
    )
    start = time.time()
    inputs_consumer, is_showings = create_table(
        view_data, consumers, movies, period,
        genre_preferences_all, view_ids_all
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
        # ラベルが2つ以上あるときに使用、ラベルは整数値で渡されることを想定
        loss='sparse_categorical_crossentropy',
        # ラベルは one-hot 表現
        # loss='categorical_crossentropy',
        metrics=[
            'accuracy',
            # Precision(),
            # Recall(),
        ],
    )

    # モデルの学習
    learning_data = inputs_consumer[:, :-1]
    # 「何も見ない」次元を追加したラベル(range=(0, n_movies))
    teaching_data = inputs_consumer[:, -1].astype(int)
    # one-hot 表現
    # teaching_data = np.eye(n_labels)[inputs_consumer[:, -1].astype(int)]
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
    # モデルチェック(period=100, n_consumers=800): 約 4 ~ 6 min
    # モデルチェック(period=400, n_consumers=800): 約 18 ~ 19 min
    start = time.time()
    label_test, label_pred, conf_matrix = check_accuracy(
        nn, movies, consumers, view_data, is_showings,
        n_params=n_params,
        genre_preferences_all=genre_preferences_all,
        view_ids_all=view_ids_all
    )
    print(
        f"Time to check accuracy(period: {period}, consumers: {n_consumers})"
        f" -> {time.time() - start:.1f} sec"
    )
    print(f"\nteaching-data:\n{inputs_consumer[:, -1].astype(int)}")

    return inputs_consumer, label_test, label_pred, conf_matrix


# TODO: インバランスデータの可能性を確認(!! 最優先 !!)
# TODO: 「映画を見た」データのオーバーサンプリングを試してみる(やり方の詳細は要検討)
def check_accuracy(
    model,
    movies: List[Movie],
    consumers: List[Consumer],
    view_data: np.ndarray,
    is_showings: np.ndarray,
    n_params: int,
    genre_preferences_all: np.ndarray,
    view_ids_all: np.ndarray
) -> Tuple[np.ndarray, List[int], tf.Tensor]:
    # パラメータ情報を取得
    n_movies, n_consumers, period = view_data.shape

    # ダミーデータを使って実際に予測してみる
    # NOTE: 現状は全てで学習した後に全てをチェック対象にしている("テスト"より"チェック")
    is_correct_predicts = defaultdict(list)
    n_missed = 0
    n_labels = n_movies + 1
    predict_ids = []
    y_pred = []
    y_test_one_hot = np.empty((1, n_labels))
    print('consumer_ID: ', end='')
    iter_check_accuracy = zip(
        consumers,
        view_ids_all,
        genre_preferences_all
    )
    for consumer, view_ids, preference in iter_check_accuracy:
        if consumer.id_num % 80 == 0:
            print(consumer.id_num, end=', ', flush=True)

        # 公開情報を選好度で上書きし、鑑賞作品 ID を付与する(重複は削除)
        preferences_showing = np.multiply(
            is_showings,
            preference
        )
        preferences_showing_with_viewed_id = remove_duplicate_rows(
            np.concatenate(
                [preferences_showing, view_ids.reshape(-1, 1)],
                axis=1,
            )
        )

        # 個人パラメータを追加して日毎に予測及び正解のチェックを行う
        params = get_params(consumer)
        n_datumn, _ = preferences_showing_with_viewed_id.shape
        datumn = np.concatenate([
            (np.ones((n_datumn, n_params)) * params).reshape(-1, n_params),
            preferences_showing_with_viewed_id,
        ], axis=1)
        view_ids_unique = datumn[:, -1].astype(int)
        view_ids_one_hot = np.eye(period)[view_ids_unique][:, :n_labels]
        y_test_one_hot = np.concatenate(
            [y_test_one_hot, view_ids_one_hot],
            axis=0
        )
        iter_viewing = zip(view_ids_unique, view_ids_one_hot, datumn[:, :-1])
        # print('mispredicted:')
        for view_id, view_correct, data in iter_viewing:
            view_predict = model.predict(data.reshape(1, -1))
            id_predict = np.argmax(view_predict)
            y_pred.append([id_predict])
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
                n_missed += 1
            #     print(
            #         f"correct-ID: {view_id}({view_genre}), "
            #         f"predict-ID: {id_predict}({genre_predict})"
            #     )

    print(f"\nmispredicted: {n_missed}\n")
    print('genre: n_histories, n_corrects')
    for genre, history in is_correct_predicts.items():
        print(f"{genre}: {len(history)}, {sum(history)}")
    predict_ids_set = set(predict_ids)
    if len(predict_ids_set) == 1:
        pred_id = predict_ids[0]
        pred_genre = pred_id if pred_id == n_movies else movies[pred_id].genre
    print(
        f"Network predicted all as same: ({pred_id}, {pred_genre})"
    )
    y_test = np.argmax(y_test_one_hot[1:, :], axis=1)
    conf_matrix = confusion_matrix(y_test, y_pred)

    return y_test, y_pred, conf_matrix


def create_table(
    view_data: np.ndarray,
    consumers: List[Consumer],
    movies: List[Movie],
    period: int,
    genre_preferences_all: np.ndarray,
    view_ids_all: np.ndarray
) -> np.ndarray:
    # 各映画の公開ラベル(size=(period, n_movies))を付与
    INTERVAL_MOVIE = 70
    is_showings = np.zeros((period, len(movies)))
    for i, movie in enumerate(movies):
        broadcast_day = movie.broadcast_day
        final_day = broadcast_day + INTERVAL_MOVIE
        is_showings[broadcast_day:final_day, i] = 1

    generator_inputs = _create_table(
        consumers=consumers,
        view_ids_all=view_ids_all,
        movies=movies,
        is_showings=is_showings,
        genre_preferences_all=genre_preferences_all
    )
    inputs = np.concatenate(
        [inputs_consumer for inputs_consumer in generator_inputs],
        axis=0
    )

    return inputs, is_showings


def _create_table(
    consumers: List[Consumer],
    view_ids_all: np.ndarray,
    movies: List[Movie],
    is_showings: np.ndarray,
    genre_preferences_all: np.ndarray
) -> Generator[np.ndarray, None, None]:
    # 鑑賞状況を踏まえた公開情報ラベルをマスク化する
    # 公開情報ラベルに消費者の鑑賞ラベルを付加する(size=(period, n_movies+1))
    iter_create_table = zip(consumers, view_ids_all, genre_preferences_all)
    for consumer, view_ids, genre_preference in iter_create_table:
        is_showings_with_viewed_id = np.concatenate(
            [is_showings, view_ids.reshape(-1, 1)],
            axis=1,
        )
        # 公開状況の組のマスク(size=(label-kinds, n_movies+1))
        # NOTE: 映画を見たかどうかも含める
        mask_showing_with_viewed_id = remove_duplicate_rows(
            is_showings_with_viewed_id
        )
        n_label_kinds = len(mask_showing_with_viewed_id)

        # 各映画に該当するジャンル選好度をマスクする(size=(label-kinds, n_movies))
        genre_preferences_masked = np.multiply(
            mask_showing_with_viewed_id[:, :-1],
            genre_preference.reshape(1, -1),
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


def remove_duplicate_rows(array: np.ndarray) -> np.ndarray:
    return np.unique(array, axis=0)


if __name__ == '__main__':
    period = int(input('enter period: '))
    main(period=period)
