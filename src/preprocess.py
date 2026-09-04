"""Inspect the preprocessing pipeline.

``python src/preprocess.py`` loads the raw data, runs feature engineering plus the
``ColumnTransformer`` and prints the resulting feature-matrix dimensions. The
transforms themselves live in :mod:`churn.features` and are refit per CV fold
inside the model pipeline rather than applied once here.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from churn.data import clean, load_raw, split  # noqa: E402
from churn.features import build_feature_pipeline  # noqa: E402


def main() -> None:
    x, y = clean(load_raw())
    x_train, x_test, y_train, _ = split(x, y)

    feature_pipe = build_feature_pipeline()
    train_matrix = feature_pipe.fit_transform(x_train, y_train)
    test_matrix = feature_pipe.transform(x_test)
    names = list(feature_pipe.named_steps["prep"].get_feature_names_out())

    print("Dataset loaded and cleaned successfully!")
    print(f"Raw feature columns    : {x.shape[1]}")
    print(f"Churn rate             : {y.mean() * 100:.1f}%")
    print(f"Training rows          : {x_train.shape[0]}")
    print(f"Testing rows           : {x_test.shape[0]}")
    print(f"Transformed dimensions : {train_matrix.shape[1]}")
    print(f"Train matrix shape     : {train_matrix.shape}")
    print(f"Test matrix shape      : {test_matrix.shape}")
    print(f"First 10 feature names : {names[:10]}")


if __name__ == "__main__":
    main()
