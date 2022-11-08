"""
市場っぽいデータを作り出す（市場やアンケート調査などによる正確なデータが手に入れられないため）
所要時間は 80 秒程度
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

import json
import matplotlib.pyplot as plt
import numpy as np
import functools
import os
import pandas as pd
import random

from collections import defaultdict
from dataclasses import dataclass
from dummy_creator import create_dummy as dummy
from functools import partial
from typing import Dict, Generator, List, Optional, Tuple, Union

from aux_simu import get_preferences_all
from models import Consumer, Movie
from const import (
    LIKE, DISLIKE, UNCONCERNED,
    DOCUMENTARY, HORROR, FANTASY, ANIME,
    SF, COMEDY, DRAMA, ACTION_ADVENTURE,
    GENRES, BROADCAST_PERIOD,
)


# TODO: 疑似的なユーザー同士のつながり（情報交換）を生成
# TODO: -> ランダムにネットワークを形成して followee の鑑賞状況を参照する？
def main(
    input_file: str,
    period: int,
    use_csv: bool = False,
    genre_movies: List[str] = None
):
    # json ファイルを利用してダミーデータを作成
    if use_csv:
        input_files = [input_file]
        dummy(params=None, input_files=input_files)
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
        # 消費者リストの取得
        consumers = get_consumers(
            preferences=preferences,
            customer_types=customer_types,
            n_initial_views=n_initial_views,
            salaries=salaries,
            does_like_movies=does_like_movies,
            children_genres=children_genres
        )

        # 映画の基本情報を読み込む
        # 公開日データ
        file_path = os.path.join(csv_directory, 'broadcast_day.csv')
        broadcast_days = np.loadtxt(file_path, skiprows=1).astype(int)
        # 宣伝費データ
        file_path = os.path.join(csv_directory, 'promotion_cost.csv')
        # NOTE: models.py では int だが、今だけ型違い
        MILLION = 1_000_000
        promotion_costs = np.loadtxt(file_path, skiprows=1) / MILLION
        min_promo_cost = promotion_costs.min()
        max_promo_cost = promotion_costs.max()
        range_promo_cost = max_promo_cost - min_promo_cost
        # ジャンルを生成
        genre_movies = random.choices(GENRES, k=len(broadcast_days))

        # NOTE* target （顧客層の狙い）は未実装、どう実装するか
        movies = get_movies(
            genre_movies=genre_movies,
            promotion_costs=promotion_costs,
            broadcast_days=broadcast_days
        )

    # ファイル読み込みをなくして高速化
    else:
        with open(input_file) as f:
            params_conf = json.load(f)
        params = dummy(params=[params_conf], input_files=None)[0]

        # 消費者の基本情報を読み込む
        # NOTE: 性別は未使用（どう使うのか？） -> 一先ず使わない
        # NOTE: status や business（家庭状況等）はどう反映させるのか
        # -> 一先ず保留

        # 性別データ
        # file_name = os.path.join(csv_directory, 'gender.csv')
        # genders = list_genders()
        # ジャンル選好度データ
        preferences = list_preferences(params)
        # 消費者カテゴリーデータ
        customer_types = list_customer_types(params=params)
        # 初期鑑賞回数データ
        key = 'n_initial_view'
        n_initial_views = np.concatenate(params[key]['rows']).astype(int)
        # 給料データ
        key = 'salary'
        salaries = np.concatenate(params[key]['rows']).astype(int)
        min_salary = salaries.min()
        max_salary = salaries.max()
        range_salary = max_salary - min_salary
        # （保護者同伴となる）子供が好きなジャンル（簡単のために1つのみ）
        key = 'children_genres'
        children_genres = np.apply_along_axis(
            label2value_children_genre,
            axis=1,
            arr=params[key]['rows']
        )
        # 映画自体が好きかどうか
        key = 'does_like_movie'
        does_like_movies = np.array(params[key]['rows'])[:, 1].astype(bool)
        # 消費者リストの取得
        consumers = get_consumers(
            preferences=preferences,
            customer_types=customer_types,
            n_initial_views=n_initial_views,
            salaries=salaries,
            does_like_movies=does_like_movies,
            children_genres=children_genres
        )

        # 映画の基本情報を読み込む
        # 公開日データ
        key = 'broadcast_day'
        broadcast_days = np.concatenate(params[key]['rows']).astype(int)
        # 宣伝費データ
        key = 'promotion_cost'
        # NOTE: models.py では int だが、今だけ型違い
        MILLION = 1_000_000
        promotion_costs = np.concatenate(
            params[key]['rows']
        ).astype(int) / MILLION
        min_promo_cost = promotion_costs.min()
        max_promo_cost = promotion_costs.max()
        range_promo_cost = max_promo_cost - min_promo_cost
        # ジャンルを生成
        if genre_movies is None:
            genre_movies = random.choices(GENRES, k=len(broadcast_days))

        # NOTE: target （顧客層の狙い）は未実装、どう実装するか
        movies = get_movies(
            genre_movies=genre_movies,
            promotion_costs=promotion_costs,
            broadcast_days=broadcast_days
        )

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
    n_movies = len(movies)
    n_consumers = len(consumers)
    view_data = [np.zeros((n_consumers, period)) for _ in range(n_movies)]
    idx = -1
    label_is_viewed_fiexed = partial(
        label_is_viewed,
        period=period,
        min_promo_cost=min_promo_cost,
        range_promo_cost=range_promo_cost,
        min_salary=min_salary,
        range_salary=range_salary
    )
    for movie, data in zip(movies, view_data):
        idx += 1

        label_is_viewed_fiexed_movie = partial(
            label_is_viewed_fiexed,
            movie=movie,
        )
        for day in range(period):
            not_viewed = np.logical_not(data.sum(axis=1).astype(bool))
            # NOTE: 日付ごとのデータは本当に必要なのか？（モデルの評価には必要だけど。。。）
            consumers_not_viewed = np.array(consumers)[not_viewed]
            past_view_data = view_data[idx][:, :day]
            label_is_viewed_fiexed_movie_day = partial(
                label_is_viewed_fiexed_movie,
                day=day,
                past_view_data=past_view_data
            )
            labels_generator = gen_label_is_viewed(
                label_is_viewed_fixed=label_is_viewed_fiexed_movie_day,
                consumers=consumers_not_viewed
            )
            data[not_viewed, day] = np.array(
                [label for label in labels_generator]
            )

    return np.array(view_data), consumers, movies


def gen_label_is_viewed(
    label_is_viewed_fixed: partial,
    consumers: List[Consumer],
) -> Generator[int, None, None]:
    for consumer in consumers:
        label = label_is_viewed_fixed(consumer=consumer)

        yield label


def calc_random_preference(interval: float = 1.0, offset: float = 0.0) -> float:
    random_preference = random.random() * interval + offset

    return random_preference


def label_is_viewed(
    consumer: Consumer,
    day: int,
    period: int,
    movie: Movie,
    min_promo_cost: int,
    range_promo_cost: int,
    min_salary: int,
    range_salary: int,
    past_view_data: np.ndarray,
) -> int:
    probability = random.random()

    # 平日（月〜金）は見に行きにくい
    if day % 7 <= 5:
        probability *= 0.7

    # 公開から時間が経つと見にくくなる
    broadcast_day = movie.broadcast_day
    elapsed_day = day - broadcast_day
    # 公開前及び公開終了後(70日経過後)は
    if elapsed_day < 0 or elapsed_day > BROADCAST_PERIOD:
        probability = 0
    else:
        # probability *= (1 - elapsed_day / (period - broadcast_day)) * 0.9
        probability *= (1 - elapsed_day / BROADCAST_PERIOD) * 0.9

    # 作品ごとの属性による確率の範囲
    assert 0 <= probability <= 1, 'invalid probability'

    # 宣伝費が高いほど観客が増えやすい
    # 全作品を通しての宣伝費のレンジに対して、
    # 各作品の宣伝費が最低額よりどの程度高いかに応じて倍率を計上する
    # NOTE: あまり効果がハッキリしない（是非は不明）
    promo_cost_level = (movie.promo_cost - min_promo_cost) / range_promo_cost
    probability *= 1 + promo_cost_level

    # 収入が高いほど観客が増えやすい
    # 全消費者の収入のレンジに対して、
    # 各消費者の収入が最低額よりどの程度高いかに応じて倍率を計上する
    salary_level = (consumer.richness - min_salary) / range_salary
    probability *= 1 + salary_level

    # 好きなジャンルほどよく見る
    genre_preference: float = consumer.genre_preference[movie.genre]
    # ジャンルを好む度合いに合わせて鑑賞確率を乗じる
    probability *= 1 + (genre_preference - 0.3)
    # 子供の好みのジャンルは見る機会が増える
    # NOTE: 現段階ではスキップ(8/30)
    # if consumer.children_genre == movie.genre:
    #     probability *= 1.3
    # フォローしている人が見ているほど見たくなる
    # NOTE: 現段階ではスキップ(8/30)
    # n_followee = len(consumer.followee)
    # if n_followee != 0:
    #     followee_view_data = past_view_data[consumer.followee, :].sum(axis=1)
    #     n_viewed_followee = followee_view_data.sum()
    #     probability *= 1 + n_viewed_followee / n_followee

    # 消費者ごとの属性等を反映して確率が1を超えた場合の調整
    if probability > 1:
        probability = 1

    # 映画が嫌いな人は映画をあまり見ない
    if not consumer.does_like_movie:
        probability *= 0.2

    # 映画を何度も見ていると見にくくなる
    probability *= (1 / 0.99) ** consumer.n_views

    if probability > 1 - genre_preference:
        label = 1
        consumer.n_views += 1
    else:
        label = 0

    return label


def list_preferences(params=None) -> List[Dict[str, float]]:
    """選好度ラベルデータを数値に変換する
    """
    if params:
        label2value_preference_apply = partial(
            np.apply_along_axis,
            func1d=lambda vec: label2value_preference(vec),
            axis=1
        )
        preferences_dict = {
            genre: label2value_preference_apply(arr=rows2arr(params, genre))
            for genre in GENRES
        }
    else:
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

    # NOTE: preference の値は手動調整
    if label_str == LIKE:
        interval = 0.25
        offset = 0.75
    elif label_str == DISLIKE:
        interval = 0.25
        offset = 0.0
    else:
        interval = 0.5
        offset = 0.25

    value = calc_random_preference(interval=interval, offset=offset)
    return value


def rows2arr(obj: List[Tuple[str, int]], genre: str) -> np.ndarray:
    genre_key = genre + '_preference'
    return np.array(obj[genre_key]['rows'])


def list_customer_types(file_name: str = None, params=None) -> List[str]:
    if params:
        customer_types = np.array(params['customer_type']['rows'])[:, 0]
    else:
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


def get_consumers(
    preferences,
    customer_types,
    n_initial_views,
    salaries,
    does_like_movies,
    children_genres
) -> List[Consumer]:
    # ID リスト
    n_consumers = len(does_like_movies)
    consumer_ids = [i for i in range(n_consumers)]
    consumers = [
        Consumer(
            id_num=consumer_id,
            genre_preference=preference,
            consume_type=customer_type,
            richness=salary,
            n_views=n_initial_view,
            does_like_movie=does_like_movie,
            children_genre=children_genre,
        )
        for(
            consumer_id,
            preference,
            customer_type,
            n_initial_view,
            salary,
            does_like_movie,
            children_genre,
        ) in zip(
            consumer_ids,
            preferences,
            customer_types,
            n_initial_views,
            salaries,
            does_like_movies,
            children_genres,
        )
    ]

    # 疑似的なユーザー間のつながりを作成
    follow_each_other(consumers)

    return consumers


def follow_each_other(users: List[Consumer]) -> None:
    """ユーザー同士のつながりを作成
    """
    follow = functools.partial(
        _follow_each_other,
        min_followee=0, max_followee=len(users)
    )

    for user in users:
        follow(user)
    return


def _follow_each_other(user, min_followee, max_followee) -> None:
    n_followee = random.randint(min_followee, max_followee // 10)

    candidats = [
        i for i in range(min_followee, max_followee)
        if i != user.id_num
    ]
    followee = random.sample(candidats, n_followee)

    user.followee = followee
    return


def get_movies(
    genre_movies,
    promotion_costs,
    broadcast_days
) -> List[Movie]:
    movies = [
        Movie(
            genre=genre,
            promo_cost=promo_cost,
            broadcast_day=broadcast_day,
        )
        for genre, promo_cost, broadcast_day
        in zip(genre_movies, promotion_costs, broadcast_days)
    ]

    return movies


if __name__ == '__main__':
    dir_name = os.path.dirname(__file__)
    input_file = os.path.normpath(os.path.join(dir_name, '../json/spec.json'))

    PERIOD = 400
    # およそ3ヶ月程度は最低でも作る
    assert PERIOD >= 100, f"PERIOD {PERIOD} is too short."

    main(input_file, PERIOD)
