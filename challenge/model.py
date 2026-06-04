"""Flight-delay prediction model for flights operating at SCL airport.

This module operationalizes the work explored in ``exploration.ipynb`` by the
Data Scientist. It exposes a single :class:`DelayModel` with the three methods
required by the challenge (``preprocess``, ``fit`` and ``predict``).

Design notes
------------
* The model is trained on the **top 10 features** identified via XGBoost feature
  importance in the notebook, and uses **class balancing** to lift the recall of
  the (minority) "delayed" class — both conclusions reached by the DS.
* ``preprocess`` reindexes the one-hot encoded features onto the fixed
  :data:`TOP_10_FEATURES` set, filling absent dummies with ``0``. This makes the
  same code path valid both for training (the full dataset) and for serving (a
  single flight that activates only a couple of categories).
* A fitted estimator is persisted to :data:`_ARTIFACT_PATH` and loaded in
  ``__init__`` so that ``predict`` works on a fresh instance without an explicit
  ``fit`` call (and so the API can serve immediately at start-up).
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

# Top 10 features selected by the DS from XGBoost feature importance. The model
# is always trained and served on exactly these columns, in this order.
TOP_10_FEATURES: list[str] = [
    "OPERA_Latin American Wings",
    "MES_7",
    "MES_10",
    "OPERA_Grupo LATAM",
    "MES_12",
    "TIPOVUELO_I",
    "MES_4",
    "MES_11",
    "OPERA_Sky Airline",
    "OPERA_Copa Air",
]

# A flight is labelled as delayed when it departs more than this many minutes
# after its scheduled time.
DELAY_THRESHOLD_MINUTES: int = 15

# Raw categorical columns one-hot encoded into the feature space.
_ONE_HOT_COLUMNS = ("OPERA", "TIPOVUELO", "MES")

# Location of the persisted, fitted estimator (kept next to this module so it is
# resolved independently of the current working directory).
_ARTIFACT_PATH = Path(__file__).resolve().parent / "model.joblib"


class DelayModel:
    """Predicts whether a flight at SCL airport will be delayed."""

    def __init__(self) -> None:
        self._model = None  # Model should be saved in this attribute.
        self._load_artifact()

    def _load_artifact(self) -> None:
        """Load a previously fitted estimator, if one has been persisted."""
        if _ARTIFACT_PATH.exists():
            self._model = joblib.load(_ARTIFACT_PATH)

    @staticmethod
    def _get_min_diff(row: pd.Series) -> float:
        """Minutes between operated (``Fecha-O``) and scheduled (``Fecha-I``)."""
        fecha_o = datetime.strptime(row["Fecha-O"], "%Y-%m-%d %H:%M:%S")
        fecha_i = datetime.strptime(row["Fecha-I"], "%Y-%m-%d %H:%M:%S")
        return (fecha_o - fecha_i).total_seconds() / 60

    def preprocess(
        self,
        data: pd.DataFrame,
        target_column: str | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame] | pd.DataFrame:
        """Prepare raw data for training or prediction.

        Args:
            data (pd.DataFrame): raw data.
            target_column (str, optional): if set, the target is returned.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: features and target.
            or
            pd.DataFrame: features.
        """
        features = pd.concat(
            [pd.get_dummies(data[col], prefix=col) for col in _ONE_HOT_COLUMNS],
            axis=1,
        )
        # Guarantee exactly the 10 expected columns (and their order), filling
        # any category absent from this batch with zeros.
        features = features.reindex(columns=TOP_10_FEATURES, fill_value=0)

        if target_column is None:
            return features

        min_diff = data.apply(self._get_min_diff, axis=1)
        target = pd.DataFrame(
            np.where(min_diff > DELAY_THRESHOLD_MINUTES, 1, 0),
            columns=[target_column],
            index=data.index,
        )
        return features, target

    def fit(self, features: pd.DataFrame, target: pd.DataFrame) -> None:
        """Fit the model with preprocessed data and persist the artifact.

        Args:
            features (pd.DataFrame): preprocessed data.
            target (pd.DataFrame): target.
        """
        y = target.iloc[:, 0] if isinstance(target, pd.DataFrame) else target

        # Class balancing lifts recall on the minority "delay" class, as the DS
        # concluded. ``class_weight='balanced'`` reproduces the notebook's
        # n_y0/n_y1 weighting ratio without hard-coding sample counts.
        self._model = LogisticRegression(class_weight="balanced", random_state=1)
        self._model.fit(features.reindex(columns=TOP_10_FEATURES, fill_value=0), y)

        joblib.dump(self._model, _ARTIFACT_PATH)

    def predict(self, features: pd.DataFrame) -> list[int]:
        """Predict delays for new flights.

        Args:
            features (pd.DataFrame): preprocessed data.

        Returns:
            (List[int]): predicted targets.
        """
        if self._model is None:
            raise RuntimeError(
                "Model is not fitted and no persisted artifact was found. "
                "Call fit() first or provide a trained model.joblib."
            )
        features = features.reindex(columns=TOP_10_FEATURES, fill_value=0)
        predictions = self._model.predict(features)
        return [int(prediction) for prediction in predictions]
