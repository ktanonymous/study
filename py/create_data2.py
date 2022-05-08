"""
市場っぽいデータを作り出す（市場やアンケート調査などによる正確なデータが手に入れられないため）
自作ライブラリ(dummy_creator)を利用
映画市場をターゲットとする
作成対象は以下の通り
    * 映画作品
        * 宣伝費
        * 動員数
            * 日ごと (or 週ごと or 合計)
        * 放映開始日
    * 消費者
        * 種別
            * ネットユーザー
            * ネットフォロワー
            * 一般消費者
        * 初期映画鑑賞回数
        * ジャンルごとの選好度
        * （消費者カテゴリー）
"""

import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import random

from collections import defaultdict
from dataclasses import dataclass
from dummy_creator import create_dummy as dummy
from typing import Dict, List, Optional

from models import Consumer, Movie
from const import (
    LIKE, DISLIKE, UNCONCERNED,
    DOCUMENTARY, HORROR, FANTASY, ANIME,
    SF, COMEDY, DRAMA, ACTION_ADVENTURE,
    GENRES,
)


# TODO: 疑似的なユーザー同士のつながり（情報交換）を生成
# TODO: -> ランダムにネットワークを形成して followee の鑑賞状況を参照する？
def main(input_file: str, period: int):
    # json ファイルを利用してダミーデータを作成
    keys = dummy(input_file)
    csv_directory = os.path.join(os.path.dirname(__file__), '../csv')

    # 消費者の基本情報を読み込む
    # NOTE: 性別は未使用（どう使うのか？） -> 一先ず使わない
    # NOTE: status や business（家庭状況等）はどう反映させるのか
    # -> 一先ず保留

    # 性別データ
    # file_name = os.path.join(csv_directory, 'gender.csv')
    # genders = list_genders()
    # ジャンル選好度データ
    preferences = list_preferences()
    # 消費者カテゴリーデータ
    customer_types = list_customer_types(file_name='customer_type.csv')
    # 初期鑑賞回数データ
    file_path = os.path.join(csv_directory, 'n_initial_view.csv')
    n_initial_views = np.loadtxt(file_path, skiprows=1).astype(int)
    # 給料データ
    file_path = os.path.join(csv_directory, 'salary.csv')
    salaries = np.loadtxt(file_path, skiprows=1)
    # （保護者同伴となる）子供が好きなジャンル（簡単のために1つのみ）
    children_genres = list_children_genre(file_name='children_genres.csv')
    # 映画自体が好きかどうか
    does_like_movies = list_does_like_movie(file_name='does_like_movie.csv')

    consumers = [
        Consumer(
            genre_preference=preference,
            consume_type=customer_type,
            richness=salarie,
            n_views=n_initial_view,
            does_like_movie=does_like_movie,
            children_genre=children_genre,
        )
        for(
            preference,
            customer_type,
            n_initial_view,
            salarie,
            does_like_movie,
            children_genre,
        ) in zip(
            preferences,
            customer_types,
            n_initial_views,
            salaries,
            does_like_movies,
            children_genres,
        )
    ]

    # 映画の基本情報を読み込む
    # 公開日データ
    file_path = os.path.join(csv_directory, 'broadcast_day.csv')
    broadcast_days = np.loadtxt(file_path, skiprows=1).astype(int)
    # 宣伝費データ
    file_path = os.path.join(csv_directory, 'promotion_cost.csv')
    promotion_costs = np.loadtxt(file_path, skiprows=1).astype(int)
    min_promo_cost = promotion_costs.min()
    max_promo_cost = promotion_costs.max()
    range_promo_cost = max_promo_cost - min_promo_cost
    # ジャンルを生成
    genre_movies = random.choices(GENRES, k=len(broadcast_days))

    # target （顧客層の狙い）は未実装、どう実装するか
    movies = [
        Movie(
            genre=genre,
            promo_cost=promo_cost,
            broadcast_day=broadcast_day,
        )
        for genre, promo_cost, broadcast_day
        in zip(genre_movies, promotion_costs, broadcast_days)
    ]

    # 鑑賞ラベルの生成
    # TODO: 鑑賞するかどうかを判定する方法の再確認
    # 映画の売り上げデータを作成
    # NOTE: 鑑賞に行けるかどうかの確率を入れても良い o
    # NOTE: 鑑賞したいかどうかの気持ちを入れても良い（時間変化も考慮）
    # NOTE: 研究室内でアンケート調査
    # NOTE: 収入が少ない場合は見れないが、収入が増えていくと飽和しそう
    # NOTE: 映画の属性にもよるのでは...（ファミリー向けには家族で行く傾向があるなど）
    # NOTE: 学習傾向に現れるかどうかは課題になる
    # 動員データフレームの作成

    # 全ての映画について
    n_customers = len(consumers)
    n_movies = len(movies)
    view_data = [np.zeros((n_customers, period)) for _ in range(n_movies)]
    n_cols = 4
    n_rows = math.ceil(n_movies / n_cols)
    fig, ax = plt.subplots(n_rows, n_cols, sharex='col', sharey='row')
    idx = -1
    for movie, data in zip(movies, view_data):
        idx += 1
        row, col = divmod(idx, n_cols)
        genre_preference = np.array([
            consumre.genre_preference[movie.genre]
            for consumre in consumers
        ])

        for day in range(period):
            not_viewed = np.logical_not(data.sum(axis=1).astype(bool))
            # NOTE: 日付ごとのデータは本当に必要なのか？（モデルの評価には必要だけど。。。）
            consumers_not_viewed = np.array(consumers)[not_viewed]
            data[not_viewed, day] = np.array(
                [
                    label_is_viewed(
                        consumer,
                        day,
                        movie.broadcast_day,
                        period,
                        movie,
                        min_promo_cost,
                        range_promo_cost,
                    )
                    for consumer in consumers_not_viewed
                ]
            )

        ax[row, col].set_title(f"{movie.genre}")
        ax[row, col].hist(data.sum(axis=1), label='view_data')
        ax[row, col].hist(genre_preference, label='preference')
        ax[row, col].legend()

    plt.show()


def calc_random_preference() -> float:
    random_preference = random.random() * 0.5

    return random_preference


def label_is_viewed(
    consumer: Consumer,
    day: int,
    broadcast_day: int,
    period: int,
    movie: Movie,
    min_promo_cost: int,
    range_promo_cost: int,
) -> int:
    probability = random.random()

    # 平日（月〜金）は見に行きにくい
    if day % 7 <= 5:
        probability *= 0.9

    # 公開から時間が経つと見にくくなる
    elapsed_day = day - broadcast_day
    if elapsed_day < 0:
        probability = 0
    else:
        probability *= (1 - elapsed_day / (period - broadcast_day)) * 0.9

    # 作品ごとの属性による確率の範囲
    assert 0 <= probability <= 1, 'invalid probability'

    # 宣伝費が高いほど観客が増えやすい
    # 全作品を通しての宣伝費のレンジに対して、
    # 各作品の宣伝費が最低額よりどの程度高いかに応じて倍率を計上する
    promo_cost_level = (movie.promo_cost - min_promo_cost) / range_promo_cost
    probability *= 1 + promo_cost_level

    # 好きなジャンルほどよく見る
    genre_preference: float = consumer.genre_preference[movie.genre]
    # ジャンルを好む度合いに合わせて鑑賞確率を乗じる
    probability *= 1 + (genre_preference - 0.3)
    # 好みを反映して確率が1を超えた場合の調整
    if probability > 1:
        probability = 1

    # 映画が嫌いな人は映画を見ない
    if not consumer.does_like_movie:
        probability = 0

    label = 1 if probability > (1 - genre_preference) else 0

    return label


def list_preferences() -> List[Dict[str, float]]:
    """csv の選好度ラベルデータを数値に変換する
    """
    directory = os.path.dirname(__file__)
    csv_directory = os.path.join(directory, '../csv')

    preferences_dict = {
        genre: _list_preferences(genre, directory=csv_directory)
        for genre in GENRES
    }

    n_consumers = len(list(preferences_dict.values())[0])
    preferences = [
        {
            genre: preferences_dict[genre][n]
            for genre in GENRES
        }
        for n in range(n_consumers)
    ]

    return preferences


def _list_preferences(genre: str, directory: str) -> List[float]:
    file_name = genre + '_preference.csv'
    file_path = os.path.join(directory, file_name)

    labels = pd.read_csv(file_path, delimiter=',')
    preference = labels.apply(label2value_preference, axis=1)

    return list(preference)


def label2value_preference(label: pd.core.series.Series) -> float:
    label_str = label[0]

    if label_str == LIKE:
        value = 0.9
    elif label_str == DISLIKE:
        value = 0.1
    else:
        value = calc_random_preference()

    return value


def list_customer_types(file_name: str) -> List[str]:
    directory = os.path.dirname(__file__)
    csv_directory = os.path.join(directory, '../csv')
    file_path = os.path.join(csv_directory, file_name)

    labels = pd.read_csv(file_path, delimiter=',')
    customer_types = labels.apply(label2value_customer_type, axis=1)

    return list(customer_types)


def label2value_customer_type(label: pd.core.series.Series) -> str:
    return label[0]


def list_does_like_movie(file_name: str) -> List[bool]:
    directory = os.path.dirname(__file__)
    csv_directory = os.path.join(directory, '../csv')
    file_path = os.path.join(csv_directory, file_name)

    labels = pd.read_csv(file_path, delimiter=',')
    does_like_movies = labels.apply(label2value_does_like_movie, axis=1)

    return list(does_like_movies)


def label2value_does_like_movie(label: pd.core.series.Series) -> bool:
    return bool(label[1])


def list_children_genre(file_name: str) -> List[Optional[str]]:
    directory = os.path.dirname(__file__)
    csv_directory = os.path.join(directory, '../csv')
    file_path = os.path.join(csv_directory, file_name)

    labels = pd.read_csv(file_path, delimiter=',')
    children_genres = labels.apply(label2value_children_genre, axis=1)

    return list(children_genres)


def label2value_children_genre(label: pd.core.series.Series) -> Optional[str]:
    value = label[0]
    if value == 'None':
        value = None

    return value


def csv2list(file_name: str, label2value) -> List[Optional[str]]:
    directory = os.path.dirname(__file__)
    csv_directory = os.path.join(directory, '../csv')
    file_path = os.path.join(csv_directory, file_name)

    labels = pd.read_csv(file_path, delimiter=',')
    does_like_movies = labels.apply(label2value, axis=1)

    return list(does_like_movies)


def label2value(label: pd.core.series.Series) -> list:
    return


if __name__ == '__main__':
    dir_name = os.path.dirname(__file__)
    input_file = os.path.normpath(os.path.join(dir_name, '../json/spec.json'))

    PERIOD = 400
    # およそ3ヶ月程度は最低でも作る
    assert PERIOD >= 100, f"PERIOD {PERIOD} is too short."

    main(input_file, PERIOD)
