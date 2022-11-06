import numpy as np

from py.predict import main as _predict


MAS_PATH = '/Users/kota/Desktop/lab/study/'

# 入力値の読み込み
file_input = MAS_PATH + 'csv/simulation/input.csv'
inputs = np.char.replace(
    np.loadtxt(file_input, delimiter=',', dtype=str)[:-1],
    '"',
    ''
).astype(float)

# 学習モデルによる予測
file_network = MAS_PATH + 'ml_models'
view_prob = _predict(inputs, file_network)

# 鑑賞確率を記録
file_output = MAS_PATH + 'csv/simulation/output.csv'
np.savetxt(file_output, view_prob, delimiter=',')
