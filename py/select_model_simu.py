import numpy as np

from tensorflow.keras import Sequential
from typing import List, NamedTuple, Tuple

from train_simu import main as _train


def main(
    period: int = 400,
    use_csv: bool = False,
    input_file: str = None,
    show_history: bool = False,
    show_time: bool = False,
    save_model: bool = True,
    save_file: str = None,
    genre_movies: List[str] = None
) -> Tuple[NamedTuple, Sequential]:
    training_obj = _train(
        period=period,
        use_csv=use_csv,
        input_file=input_file,
        show_history=show_history,
        show_time=show_time,
        genre_movies=genre_movies
    )

    # モデルのテスト及び最高精度のモデル選択
    test_data = training_obj.test_data
    test_target = training_obj.test_target
    all_pred_ids = training_obj.all_pred_ids
    n_test_data, _ = test_data.shape
    test_accs = np.apply_along_axis(
        lambda vec: (vec == test_target).sum() / n_test_data,
        axis=1,
        arr=all_pred_ids
    )
    best_model_idx = np.argmax(test_accs)
    best_model = training_obj.models[best_model_idx]

    # 最高精度モデルの保存
    if save_file is None:
        save_file = '../ml_models'
    best_model.save(save_file)

    return training_obj, best_model


if __name__ == '__main__':
    main()
