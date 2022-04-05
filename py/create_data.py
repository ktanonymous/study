"""
市場っぽいデータを作り出す（市場やアンケート調査などによる正確なデータが手に入れられないため）
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
import pandas as pd
import random

from dataclasses import dataclass
from typing import Dict, List, Optional


# ダミーデータ生成ライブラリ faker/mimesis について
# NOTE: mimesis の方が良さそう？ v.5.1.0 は不安定、v.5.0.0 は安定版っぽい？(2021/11/21)
# NOTE: mimesis のリファレンス(https://mimesis.name/api.html#field)
# https://exture-ri.com/2018/11/07/python%E3%82%92%E4%BD%BF%E3%81%A3%E3%81%9F%E3%83%80%E3%83%9F%E3%83%BC%E3%83%87%E3%83%BC%E3%82%BF%E7%94%9F%E6%88%90/
# https://www.nblog09.com/w/2019/01/24/python-faker/
# https://wonderwall.hatenablog.com/entry/2017/08/23/223000

@dataclass
class Consumer(object):
    genre_preference: Dict[str, List[float]]
    status: str
    motivation: float
    consume_type: str
    can_view: int
    richness: float
    busyness: float


@dataclass
class Movie(object):
    genre: str
    target: str
    promo_cost: int
    bloadcast_day: int


def main(period: int, n_movies: int, n_people: int):
    # 人々のデータを作成
    RATIO_NET_USER = 0.2
    RATIO_NET_FOLLOWER = 0.3
    RATIO_CONSUMER = 0.5
    n_net_users = int(n_people * RATIO_NET_USER)
    n_net_followers = int(n_people * RATIO_NET_FOLLOWER)
    n_consumers = int(n_people * RATIO_CONSUMER)
    # 初期作品鑑賞回数を対数正規分布(https://sci-fx.net/math-log-norm-dist/)で初期化
    MEDIAN_INITIAL_N_VIEW_MOVIES = 9
    MODE_INITIAL_N_VIEW_MOVIES = 8
    mu = math.log((MEDIAN_INITIAL_N_VIEW_MOVIES))
    sigma = math.sqrt(mu - math.log(MODE_INITIAL_N_VIEW_MOVIES))
    initial_n_view_movies = np.random.lognormal(
        mu, sigma, n_net_followers + n_consumers
    ).astype(int)
    # 消費者データの作成
    n_customers = n_net_followers + n_consumers
    # ジャンル選好度の作成
    DOCUMENTARY = 'documentary'
    HORROR = 'horror'
    FANTASY = 'fantasy'
    ANIME = 'anime'
    SF = 'sf'
    COMEDY = 'comedy'
    DRAMA = 'drama'
    ACTION_ADVENTURE = 'action_adventure'
    genres = [
        DOCUMENTARY, HORROR, FANTASY, ANIME,
        SF, COMEDY, DRAMA, ACTION_ADVENTURE,
    ]
    genre_preferences = {
        genre: gen_random_genre_preference(genre=genre, n_gen=n_customers)
        for genre in genres
    }
    # 消費者カテゴリー情報を作成
    CUSTOMER_TYPE_RATIO = {
        'initial_user': 0.16,
        'pre_user': 0.34,
        'after_user': 0.34,
        'late_user': 0.16,
    }
    customer_type = random.choices(
        list(CUSTOMER_TYPE_RATIO.keys()),
        k=n_customers,
        weights=CUSTOMER_TYPE_RATIO.values(),
    )

    # 映画のデータを作成
    MIN_PROMOTION_COST = 1 * (10 ** 7)
    MAX_PROMOTION_COST = 10 * (10 ** 7)
    promotion_cost_range = (MIN_PROMOTION_COST, MAX_PROMOTION_COST)
    broadcast_range = [0, period-100]
    # NOTE: 端数が割と気持ち悪いかもしれない
    promotion_costs = np.array(
        [random.randint(*promotion_cost_range) for _ in range(n_movies)]
    )
    broadcast_days = np.array(
        [random.randint(*broadcast_range) for _ in range(n_movies)]
    )
    genre_movies = dict(enumerate(random.choices(genres, k=n_movies)))

    # NOTE: 鑑賞に行けるかどうかの確率を入れても良い o
    # NOTE: 鑑賞したいかどうかの気持ちを入れても良い（時間変化も考慮）
    # TODO: 映画の宣伝費を考慮する
    # TODO: 映画及び人の属性情報を付与する(faker/mimesis のようなライブラリを使っても良さそう)
    # TODO: 研究室内でアンケート調査
    # NOTE: そもそも映画を好むかどうかの情報も必要
    # NOTE: 家族がいる場合は家族の好みに合わせるかもしれない
    # NOTE: 個人ごとに（時間と）お金の制約がリアルには考えられるはず（忙しさ、貧富 <-- 社会的地位に依存）
    # NOTE: 収入が少ない場合は見れないが、収入が増えていくと飽和しそう
    # NOTE: 映画の特性によるのでは...（ファミリー向けには家族で行く傾向があるなど）
    # NOTE: 学習傾向に現れるかどうかは課題になる
    # 動員データフレームの作成

    # 全ての映画について
    view_data = [np.zeros((n_customers, period)) for _ in range(n_movies)]
    n_cols = 4
    n_rows = math.ceil(n_movies / n_cols)
    fig, ax = plt.subplots(n_rows, n_cols, sharex='col', sharey='row')
    idx = -1
    for genre_movie, broadcast_day, data in zip(genre_movies.values(), broadcast_days, view_data):
        idx += 1
        row, col = divmod(idx, n_cols)
        genre_preference = genre_preferences[genre_movie]

        for day in range(period):
            not_viewed = np.logical_not(data.sum(axis=1).astype(bool))
            # TODO: 日付ごとのデータは本当に必要なのか？（モデルの評価には必要だけど。。。）
            preference_not_viewed = genre_preference[not_viewed]
            data[not_viewed, day] = np.array(
                [
                    label_is_viewed(preference, day, broadcast_day, period)
                    for preference in preference_not_viewed
                ]
            )

        ax[row, col].set_title(f"{genre_movie}")
        ax[row, col].hist(data.sum(axis=1), label='view_data')
        ax[row, col].hist(genre_preference, label='preference')
        ax[row, col].legend()

    plt.show()


def calc_random_preference() -> float:
    random_preference = random.random() * 0.5

    return random_preference


def gen_random_genre_preference(genre: str, n_gen: int = 0) -> List[float]:
    genre_preference_ratio = {
        'documentary': 0.2,
        'horror': 0.27,
        'fantasy': 0.43,
        'anime': 0.45,
        'sf': 0.48,
        'comedy': 0.6,
        'drama': 0.61,
        'action_adventure': 0.69,
    }

    genre_preferences = []
    for _ in range(n_gen):
        rand = random.random()

        if rand < genre_preference_ratio[genre]:
            genre_preferences.append(0.9)
        elif rand > 0.9:
            # 10 % がジャンルを嫌うこととする
            genre_preferences.append(0.1)
        else:
            genre_preferences.append(calc_random_preference())

    return np.array(genre_preferences)


def label_is_viewed(preference: float, day: int, broadcast_day: int, period: int) -> int:
    probability = random.random()

    # 平日（月〜金）は見に行きにくい
    if day % 7 <= 5:
        probability *= 0.1

    elapsed_day = day - broadcast_day
    if elapsed_day < 0:
        probability = 0
    else:
        probability *= (1 - elapsed_day / (period - broadcast_day)) * 0.9

    assert 0 <= probability <= 1, 'invalid probability'

    label = 1 if probability > (1 - preference) else 0

    return label


if __name__ == '__main__':
    PERIOD = 400
    N_MOVIES = 18
    N_PEOPLE = 1000

    # およそ3ヶ月程度は最低でも作る
    assert PERIOD >= 100, f"PERIOD {PERIOD} is too short."

    main(period=PERIOD, n_movies=N_MOVIES, n_people=N_PEOPLE)
