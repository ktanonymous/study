import numpy as np
import random

from py.select_model import main as train


MAS_PATH = '/Users/kota/Desktop/lab/study/'
ID_GENRE = {
    1: 'documentary',
    2: 'horror',
    3: 'fantasy',
    4: 'anime',
    5: 'sf',
    6: 'comedy',
    7: 'drama',
    8: 'action_adventure'
}

# シミュレーション情報の取得
file_simu_info = MAS_PATH + 'cell_market/Environment.csv'
simu_info = dict(
    np.char.replace(
        np.loadtxt(file_simu_info, delimiter=',', dtype=str)[:, :-1],
        '"',
        ''
    )
)
period = int(simu_info['period'])
# シミュレーションと同じ映画のジャンルを利用する場合
# file_genre_id = MAS_PATH + 'data/Genre_Data.csv'
# genre_movies_id = np.loadtxt(file_genre_id, delimiter=',', dtype=int)
# genre_movies = [ID_GENRE[genre_id] for genre_id in genre_movies_id]
# 各ジャンルを同数にする場合
genres = list(ID_GENRE.values())
genre_movies = genres + random.choices(genres, k=10)

# 学習実行
file_input = MAS_PATH + 'json/spec.json'
file_save = MAS_PATH + 'ml_models'
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
