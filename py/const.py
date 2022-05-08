# 消費者カテゴリー情報を作成
CUSTOMER_TYPE_RATIO = {
    'initial_user': 0.16,
    'pre_user': 0.34,
    'after_user': 0.34,
    'late_user': 0.16,
}

# 宣伝費の最大/小値
MIN_PROMOTION_COST = 1 * (10 ** 7)
MAX_PROMOTION_COST = 10 * (10 ** 7)

# ユーザー種の比率
RATIO_NET_USER = 0.2
RATIO_NET_FOLLOWER = 0.3
RATIO_CONSUMER = 0.5

# 対数正規分布の中央値と最頻値
MEDIAN_INITIAL_N_VIEW_MOVIES = 9
MODE_INITIAL_N_VIEW_MOVIES = 8

# ジャンル
DOCUMENTARY = 'documentary'
HORROR = 'horror'
FANTASY = 'fantasy'
ANIME = 'anime'
SF = 'sf'
COMEDY = 'comedy'
DRAMA = 'drama'
ACTION_ADVENTURE = 'action_adventure'
GENRES = [
    DOCUMENTARY, HORROR, FANTASY, ANIME,
    SF, COMEDY, DRAMA, ACTION_ADVENTURE,
]

# 選好度ラベル
LIKE = 'like'
DISLIKE = 'dislike'
UNCONCERNED = 'unconcerned'
