"""2層のネットワークを利用して,
ダミーデータを教師データとした映画鑑賞に関する行動予測を行う.
"""
import numpy as np
import os
import tensorflow as tf
import time

from copy import deepcopy
from sklearn.model_selection import StratifiedKFold, train_test_split
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from typing import List, Generator, NamedTuple, Tuple

from .aux import build_model, get_preferences_all, params2labels
from .const import N_FOLDS
from .create_data_no_plot import main as create_data
from .models import Consumer, Movie


class TrainingObject(NamedTuple):
    inputs: np.ndarray
    train_data: np.ndarray
    test_data: np.ndarray
    train_target: np.ndarray
    test_target: np.ndarray
    models: List[Sequential]
    all_train_loss: List[List[float]]
    all_val_loss: List[List[float]]
    all_train_acc: List[List[float]]
    all_val_acc: List[List[float]]
    all_pred_ids: np.ndarray


def main(
    period: int = 100,
    use_csv: bool = False,
    input_file: str = None,
    show_history: bool = False,
    show_time: bool = False,
    genre_movies: List[str] = None
) -> Tuple[np.ndarray, np.ndarray, List[int], tf.Tensor]:
    if input_file is None:
        dir_name = os.path.dirname(__file__)
        input_file = os.path.normpath(
            os.path.join(dir_name, '../json/spec.json')
        )
    # データ生成(period=100, n_consumers=800): 約 5 ~ 8 sec
    # データ生成(period=400, n_consumers=800): 約 14 ~ 20 sec
    start = time.time()
    view_data, consumers, movies = create_data(
        input_file, period, use_csv, genre_movies
    )
    elapsed_time = time.time() - start
    n_consumers = len(consumers)
    genre_preferences_all = get_preferences_all(consumers, movies)
    if show_time:
        print(
            f"Time to create data(period: {period}, "
            f"consumers: {n_consumers}) -> "
            f"{elapsed_time:.1f} sec"
        )
    # テーブル作成(period=100, n_consumers=800): 約 0.5 sec
    # テーブル作成(period=400, n_consumers=800): 約 2 sec
    # 鑑賞作品 ID (size=(n_consumers, period))
    view_ids_all = np.apply_along_axis(
        func1d=np.argmax,
        axis=2,
        arr=view_data.transpose(1, 2, 0),
    )
    start = time.time()
    inputs_consumer, is_showings = create_table(
        view_data, consumers, movies, period,
        genre_preferences_all, view_ids_all
    )
    if show_time:
        print(
            f"Time to create table(period: {period}, "
            f"consumers: {n_consumers}) -> "
            f"{time.time() - start:.1f} sec"
        )
    n_movies = len(movies)
    n_params = inputs_consumer[:, :-(n_movies+1)].shape[-1]

    # パラメータのカテゴリカライズ
    n_labels = 3
    inputs_consumer_labels = deepcopy(inputs_consumer)
    params2labels(inputs_consumer_labels, n_params, n_labels)

    # 「何も見ていない」データの除外
    has_viewed = inputs_consumer_labels[:, -1] != n_movies
    inputs_consumer_labels_viewed = inputs_consumer_labels[has_viewed]
    learning_data = inputs_consumer_labels_viewed[:, :-1]
    teaching_data = inputs_consumer_labels_viewed[:, -1].astype(int)

    # 訓練データとテストデータを分割する
    train_data, test_data, train_target, test_target = train_test_split(
        learning_data, teaching_data,
        test_size=0.3,  # default = (1 - train_size) or 0.25
        # train_size=0.7,
        # random_state=88,
        shuffle=True,  # default = True
    )

    # クロスバリデーション
    kfold = StratifiedKFold(
        n_splits=N_FOLDS,  # default = 5
        shuffle=False,  # default = False
        # random_state=88,
    )
    folds = kfold.split(train_data, train_target)
    input_shape = train_data[0].shape
    models = []
    batch_size = 128
    epochs = 200
    callbacks = [
        EarlyStopping(
            monitor='loss',
            patience=10,
            restore_best_weights=True
        )
    ]
    all_train_loss = []
    all_val_loss = []
    all_train_acc = []
    all_val_acc = []
    for fold, (indices_train, indices_val) in enumerate(folds):
        model = build_model(
            input_shape=input_shape,
            output_shape=n_movies,
            name=f"movie_model_{fold}"
        )
        models.append(model)
        # 学習の実行
        learning_train = learning_data[indices_train]
        learning_val = learning_data[indices_val]
        teaching_train = teaching_data[indices_train]
        teaching_val = teaching_data[indices_val]
        history = model.fit(
            learning_train,
            teaching_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=[learning_val, teaching_val],
            verbose=show_history,
            callbacks=callbacks
        )
        # 学習結果の取得
        train_loss = history.history['loss']
        val_loss = history.history['val_loss']
        train_acc = history.history['acc']
        val_acc = history.history['val_acc']
        # 学習結果の保存
        all_train_loss.append(train_loss)
        all_val_loss.append(val_loss)
        all_train_acc.append(train_acc)
        all_val_acc.append(val_acc)

    all_pred_ids = np.array([
        np.argmax(model.predict(test_data), axis=1)
        for model in models
    ])

    return TrainingObject(
        inputs=inputs_consumer_labels_viewed,
        train_data=train_data,
        test_data=test_data,
        train_target=train_target,
        models=models,
        test_target=test_target,
        all_train_loss=all_train_loss,
        all_val_loss=all_val_loss,
        all_train_acc=all_train_acc,
        all_val_acc=all_val_acc,
        all_pred_ids=all_pred_ids
    )


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
