import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pathlib
import random

from functools import partial
from tensorflow.keras import Sequential
from typing import List, NamedTuple, Tuple

from const import GENRES
from select_model_simu import main as train


def main(n_consumers: int = 800, period: int = 400):
    work_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..')
    )

    # 消費者パラメータの生成
    gen_category_thresholds(n_consumers, work_dir)
    gen_effectivenesses(n_consumers, work_dir)
    gen_genre_preferences(n_consumers, work_dir)
    gen_initial_views(n_consumers, work_dir)

    # 学習
    n_movies = 18
    training_obj, best_model = train_network(period, work_dir, n_movies)
    save_train_history(training_obj, work_dir)

    # 鑑賞確率予測
    predict_all(best_model, work_dir, period, n_consumers)


def gen_genre_preferences(n_consumers: int, work_dir: str):
    # ジャンル選好度重みの読み込み
    file_genre_ratio = work_dir + '/data/genre_ratio.csv'
    genre_ratios = np.loadtxt(file_genre_ratio, delimiter=',', dtype=float)

    # 各消費者のジャンル選好度を生成する
    get_genre_preferences = np.frompyfunc(
        _get_genre_preferences,
        nin=1, nout=1
    )
    genre_preferences = get_genre_preferences(
        np.tile(genre_ratios, (n_consumers, 1))
    ).astype(float)

    file_genre_preferences = work_dir + '/data/genre_preferences.csv'
    np.savetxt(file_genre_preferences, genre_preferences, delimiter=',')


def _get_genre_preferences(ratio: float) -> float:
    preferencelike = 1.0
    preferencedislike = 0.0
    preferenceflat = random.random() * 0.8 + 0.1
    weights = [ratio, 0.1, 0.9 - ratio]

    genre_weight = random.choices(
        [preferencelike, preferencedislike, preferenceflat],
        k=1,
        weights=weights
    )

    return genre_weight[0]


def gen_category_thresholds(n_consumers: int, work_dir: str):
    # 各消費者タイプを生成
    _gen_category_threshold = np.frompyfunc(
        get_category_threshold,
        nin=1, nout=1
    )
    category_thresholds = _gen_category_threshold(np.random.rand(n_consumers))

    file_category_thresholds = work_dir + '/data/category_thresholds.csv'
    np.savetxt(
        file_category_thresholds,
        category_thresholds.astype(float),
        delimiter=','
    )


def get_category_threshold(x: float) -> float:
    thresholds = [
        _threshold_initial_consumer,
        _threshold_pre_consumer,
        _threshold_after_consumer,
        _threshold_late_consumer
    ]
    weights = [0.16, 0.34, 0.34, 0.16]

    threshold = random.choices(thresholds, k=1, weights=weights)[0]

    return threshold(x)


def _threshold_initial_consumer(x: float) -> float:
    return x * 50 + 940


def _threshold_pre_consumer(x: float) -> float:
    return x * 50 + 990


def _threshold_after_consumer(x: float) -> float:
    return x * 100 + 1500


def _threshold_late_consumer(x: float) -> float:
    return x * 100 + 2100


def gen_effectivenesses(n_consumers: int, work_dir):
    effectiveness_all = np.random.uniform(0.7, 1.7, (n_consumers,))
    file_effectiveness = work_dir + '/data/effectiveness.csv'
    np.savetxt(file_effectiveness, effectiveness_all, delimiter=',')


def gen_initial_views(
    n_consumers: int, work_dir: str, mode: int = 8, median: int = 9
):
    assert median > mode

    mu = math.log(median)
    sigma = math.sqrt(mu - math.log(mode))
    n_initial_view_all = np.random.lognormal(mu, sigma, (n_consumers,))

    file_initial_views = work_dir + '/data/initial_views.csv'
    np.savetxt(file_initial_views, n_initial_view_all, delimiter=',')


def train_network(
    period: int, work_dir: str, n_movies: int
) -> Tuple[NamedTuple, Sequential]:
    file_input = work_dir + '/json/spec.json'
    file_save = work_dir + '/ml_models'
    n_genres = len(GENRES)
    genre_movies = GENRES + random.choices(GENRES, k=n_movies-n_genres)

    training_obj, best_model = train(
        period=period,
        use_csv=False,
        input_file=file_input,
        show_history=False,
        show_time=False,
        save_model=True,
        save_file=file_save,
        genre_movies=genre_movies
    )

    return training_obj, best_model


def save_train_history(obj: NamedTuple, work_dir: str):
    plot_history = partial(_plot_history, models=obj.models)
    # loss の記録
    plot_history(
        all_train=obj.all_train_loss,
        all_val=obj.all_val_loss,
        var_name='Loss',
        save_file=work_dir + '/figure/loss.png'
    )

    # accuracy の記録
    plot_history(
        all_train=obj.all_train_acc,
        all_val=obj.all_val_acc,
        var_name='Accuracy',
        save_file=work_dir + '/figure/accuracy.png',
    )


def _plot_history(
    models: List[Sequential],
    all_train: List[List[float]],
    all_val: List[List[float]],
    var_name: str,
    save_file: str,
):
    plt.figure()
    plt.title(f"{var_name} of each model")
    epochs = math.ceil(len(all_train[0]) / 100) * 100
    plt.xlim(0, epochs+1)
    plt.grid()
    plt.xlabel('Epoch')
    plt.ylabel(var_name)
    for model, train, val in zip(models, all_train, all_val):
        n_history = len(train)
        epoch = list(range(1, n_history+1))
        plt.plot(epoch, train, label=f"Training data({model.name})")
        plt.plot(epoch, val, label=f"Validation data({model.name})")
    plt.legend()

    plt.savefig(save_file)


def predict_all(
    model: Sequential,
    work_dir: str,
    period: int,
    n_consumers: int
):
    # 公開日情報の取得
    file_broadcast_days = work_dir + '/data/Date_Data.csv'
    broadcast_days = np.loadtxt(file_broadcast_days, delimiter=',', dtype=int)
    n_movies, = broadcast_days.shape
    is_showings = np.zeros((period, n_movies))
    for i, broadcast_day in enumerate(broadcast_days):
        is_showings[broadcast_day:, i] = 1

    # 各消費者の映画ごとに対応するジャンル選好度を取得する
    file_genre_preferences = work_dir + '/data/genre_preferences.csv'
    genre_preferences = np.loadtxt(file_genre_preferences, delimiter=',')
    file_genre_ids = work_dir + '/data/Genre_Data.csv'
    genre_ids = np.loadtxt(file_genre_ids, delimiter=',', dtype=int)
    id2preference = partial(
        _id2preference,
        genre_ids=genre_ids
    )
    preferences = np.apply_along_axis(
        id2preference,
        axis=1,
        arr=genre_preferences
    )

    # 各消費者の映画ごとに対応するジャンル選好度を公開日情報でマスクする
    mask_preferences = partial(
        _mask_prefernce,
        mask=is_showings
    )
    preferences_showing_all = np.apply_along_axis(
        mask_preferences,
        axis=1,
        arr=preferences
    )

    # 消費者の属性値を付加する
    file_params = work_dir + '/data/initial_views.csv'
    initial_views = np.loadtxt(file_params, delimiter=',')
    max_initial_view = initial_views.max()
    params = initial_views / max_initial_view
    inputs_all = np.concatenate(
        [
            np.tile(params, (400, 1, 1)).transpose(2, 0, 1),
            preferences_showing_all
        ],
        axis=2
    )

    # 消費者ごとに、全日程の鑑賞確率予測を記録する
    save_dir = work_dir + '/data/view_probs'
    os.makedirs(save_dir, exist_ok=True)
    for idx, inputs in enumerate(inputs_all):
        view_probs = model.predict(inputs)
        save_file = save_dir + f"/consumer_{idx}.csv"
        pathlib.Path(save_file).touch(exist_ok=True)
        np.savetxt(save_file, view_probs, delimiter=',')


def _id2preference(
    genre_preference: np.ndarray,
    genre_ids: np.ndarray
) -> float:
    preference = np.array([
        genre_preference[genre_id - 1]
        for genre_id in genre_ids
    ])

    return preference


def _mask_prefernce(
    preference: np.ndarray,
    mask: np.ndarray
) -> np.ndarray:
    preference_masked = np.multiply(mask, preference)

    return preference_masked


if __name__ == '__main__':
    main()
