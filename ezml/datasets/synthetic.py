from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd


_DISTRIBUTION_SAMPLERS = {
    "normal": lambda rng, size, params: rng.normal(
        loc=params.get("loc", 0.0), scale=params.get("scale", 1.0), size=size
    ),
    "uniform": lambda rng, size, params: rng.uniform(
        low=params.get("low", 0.0), high=params.get("high", 1.0), size=size
    ),
    "exponential": lambda rng, size, params: rng.exponential(
        scale=params.get("scale", 1.0), size=size
    ),
    "lognormal": lambda rng, size, params: rng.lognormal(
        mean=params.get("mean", 0.0), sigma=params.get("sigma", 1.0), size=size
    ),
    "poisson": lambda rng, size, params: rng.poisson(
        lam=params.get("lam", 3.0), size=size
    ),
    "binomial": lambda rng, size, params: rng.binomial(
        n=params.get("n", 10), p=params.get("p", 0.5), size=size
    ),
    "gamma": lambda rng, size, params: rng.gamma(
        shape=params.get("shape", 2.0), scale=params.get("scale", 1.0), size=size
    ),
    "beta": lambda rng, size, params: rng.beta(
        a=params.get("a", 2.0), b=params.get("b", 5.0), size=size
    ),
    "chisquare": lambda rng, size, params: rng.chisquare(
        df=params.get("df", 3.0), size=size
    ),
    "triangular": lambda rng, size, params: rng.triangular(
        left=params.get("left", 0.0),
        mode=params.get("mode", 0.5),
        right=params.get("right", 1.0),
        size=size,
    ),
}


@dataclass
class SyntheticDatasetGenerator:
    """Composable synthetic tabular data generator."""

    n_samples: int = 1000
    random_state: Optional[int] = 42

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.random_state)

    def from_distributions(
        self,
        schema: Dict[str, Dict[str, object]],
        add_row_id: bool = False,
    ) -> pd.DataFrame:
        """
        Generate features from a distribution schema.

        Parameters
        ----------
        schema : dict
            Format: {
              "feature_name": {
                  "distribution": "normal",
                  ...distribution params...
              }
            }
        add_row_id : bool
            Add `row_id` column if True.
        """

        data = {}
        for feature_name, feature_spec in schema.items():
            distribution = str(feature_spec.get("distribution", "normal")).lower()
            if distribution not in _DISTRIBUTION_SAMPLERS:
                available = ", ".join(sorted(_DISTRIBUTION_SAMPLERS))
                raise ValueError(
                    f"Unsupported distribution '{distribution}' for '{feature_name}'. "
                    f"Available: {available}"
                )

            params = {
                key: value
                for key, value in feature_spec.items()
                if key != "distribution"
            }
            data[feature_name] = _DISTRIBUTION_SAMPLERS[distribution](
                self._rng, self.n_samples, params
            )

        df = pd.DataFrame(data)
        if add_row_id:
            df.insert(0, "row_id", np.arange(self.n_samples))
        return df

    def add_mathematical_features(
        self,
        df: pd.DataFrame,
        source_columns: Optional[Iterable[str]] = None,
        degree: int = 2,
        include_interactions: bool = True,
        include_trig: bool = True,
    ) -> pd.DataFrame:
        """Add polynomial, interaction, and trigonometric features."""

        if source_columns is None:
            source_columns = [
                col
                for col in df.columns
                if pd.api.types.is_numeric_dtype(df[col])
            ]
        source_columns = list(source_columns)

        out = df.copy()

        for col in source_columns:
            values = out[col].to_numpy()
            for power in range(2, degree + 1):
                out[f"{col}__pow_{power}"] = np.power(values, power)
            if include_trig:
                out[f"{col}__sin"] = np.sin(values)
                out[f"{col}__cos"] = np.cos(values)

        if include_interactions:
            for i, left in enumerate(source_columns):
                for right in source_columns[i + 1 :]:
                    out[f"{left}__x__{right}"] = out[left] * out[right]

        return out

    def add_target(
        self,
        df: pd.DataFrame,
        task: str = "regression",
        formula: Optional[Dict[str, float]] = None,
        noise: float = 0.1,
        target_name: str = "target",
        threshold: Optional[float] = None,
    ) -> pd.DataFrame:
        """Create a synthetic target from weighted feature combinations."""

        numeric_cols = [
            col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])
        ]
        if not numeric_cols:
            raise ValueError("Target generation requires numeric columns.")

        weights = formula or {
            col: float(self._rng.uniform(0.2, 1.2)) for col in numeric_cols[: min(6, len(numeric_cols))]
        }

        y = np.zeros(self.n_samples)
        for col, coef in weights.items():
            if col not in df.columns:
                raise ValueError(f"Column '{col}' not found for formula-based target generation.")
            y += coef * df[col].to_numpy()

        y += self._rng.normal(loc=0.0, scale=max(noise, 1e-8), size=self.n_samples)

        out = df.copy()
        if task == "regression":
            out[target_name] = y
        elif task == "classification":
            split = float(np.median(y) if threshold is None else threshold)
            out[target_name] = (y >= split).astype(int)
        else:
            raise ValueError("task must be either 'regression' or 'classification'.")

        return out


def make_distribution_data(
    n_samples: int = 1000,
    schema: Optional[Dict[str, Dict[str, object]]] = None,
    random_state: Optional[int] = 42,
    add_row_id: bool = False,
) -> pd.DataFrame:
    """Convenience API for distribution-driven synthetic dataset creation."""

    default_schema = {
        "normal_feature": {"distribution": "normal", "loc": 0, "scale": 1},
        "uniform_feature": {"distribution": "uniform", "low": -5, "high": 5},
        "gamma_feature": {"distribution": "gamma", "shape": 2.5, "scale": 1.2},
        "poisson_feature": {"distribution": "poisson", "lam": 3.5},
    }
    generator = SyntheticDatasetGenerator(
        n_samples=n_samples,
        random_state=random_state,
    )
    return generator.from_distributions(schema or default_schema, add_row_id=add_row_id)


def make_mathematical_synthetic_data(
    n_samples: int = 1000,
    random_state: Optional[int] = 42,
    include_target: bool = True,
    task: str = "regression",
    target_name: str = "target",
) -> pd.DataFrame:
    """Create rich synthetic data with distributions + mathematical feature expansion."""

    generator = SyntheticDatasetGenerator(
        n_samples=n_samples,
        random_state=random_state,
    )
    base = generator.from_distributions(
        {
            "x1": {"distribution": "normal", "loc": 0, "scale": 1},
            "x2": {"distribution": "uniform", "low": -3, "high": 3},
            "x3": {"distribution": "exponential", "scale": 1.5},
            "x4": {"distribution": "beta", "a": 2, "b": 6},
        }
    )
    rich = generator.add_mathematical_features(base, degree=3)

    if include_target:
        rich = generator.add_target(
            rich,
            task=task,
            formula={"x1": 1.5, "x2": -2.0, "x1__x__x2": 0.8, "x3__pow_2": 0.4},
            noise=0.3,
            target_name=target_name,
        )

    return rich


def list_supported_distributions() -> List[str]:
    """Return all supported distribution names."""

    return sorted(_DISTRIBUTION_SAMPLERS.keys())
