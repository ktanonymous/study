import numpy as np

from tensorflow.keras.models import load_model
from typing import List


def main(inputs: List[float], load_file: str = None) -> np.ndarray:
    # モデルの読み込み
    if load_file is None:
        load_file = '../ml_models'
    model = load_model(load_file)

    # 鑑賞確率の予測
    inputs = np.array(inputs)
    inputs_dim = inputs.ndim
    if inputs_dim == 1:
        inputs = inputs.reshape(1, -1)
    view_probs = model.predict(inputs)

    return view_probs


if __name__ == '__main__':
    main()
