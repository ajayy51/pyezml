import os
import pandas as pd
from sklearn.model_selection import train_test_split

from .detection import detect_task
from .models import get_model
from .preprocessing import Preprocessor
from .persistence import save_object, load_object
from .helpers import _normalize_save_path

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    r2_score,
    mean_absolute_error,
)


class AutoModel:
    """
    Beginner-friendly AutoML for tabular data.
    """

    VERSION = "1.5"

    # ================= INIT =================
    def __init__(
        self,
        task="auto",
        mode="fast",
        preprocess=True,
        scale=False,
        verbose=True,
        random_state=42,
        save=None
        
    ):
        self.task = task
        self.mode = mode
        self.use_preprocess = preprocess
        self.scale = scale
        self.verbose = verbose
        self.random_state = random_state
        self.auto_save_path = _normalize_save_path(save)
        

        self.model = None
        self.columns = None
        self.trained = False
        self.preprocessor = None
        self.metrics_ = None

    # ================= TRAIN =================
    def train(self, data, target, save=None):
        # ---------------- LOAD DATA ----------------
        import pandas as pd
        import os

# Case 1: user passed DataFrame
        if isinstance(data, pd.DataFrame):
            df = data.copy()

# Case 2: user passed CSV path
        elif isinstance(data, str):
            if not os.path.exists(data):
                raise FileNotFoundError(f"File not found: {data}")
            df = pd.read_csv(data)

        else:
            raise ValueError(
                "data must be a pandas DataFrame or CSV file path."
            )

# ---------------- BASIC VALIDATION ----------------

        if df.empty:
            raise ValueError("Dataset is empty.")

        df.columns = df.columns.str.strip()

        if target not in df.columns:
            raise ValueError(f"Target column '{target}' not found.")

        X = df.drop(columns=[target])
        y = df[target]

        if X.shape[1] == 0:
            raise ValueError("No feature columns found.")

        self.columns = X.columns.tolist()

        # ===== task detection =====
        if self.task == "auto":
            self.task = detect_task(y)
            if self.verbose:
                print("Auto task detection used")

        if self.verbose:
            print(f"Detected task: {self.task}")

        # ===== preprocessing =====
        if self.use_preprocess:
            if self.verbose:
                print("Preprocessing data...")
            self.preprocessor = Preprocessor(scale=self.scale)
            X = self.preprocessor.fit_transform(X)

        # ---------------- MODEL SELECTION ----------------

        effective_mode = self.mode

        # small-data guard for LightGBM
        if self.mode == "best" and len(X) < 200:
            if self.verbose:
                print("Dataset too small for LightGBM. Using RandomForest instead.")
            effective_mode = "fast"

        self.model = get_model(
            task=self.task,
            mode=effective_mode,
            random_state=self.random_state,
        )

        if self.model is None:
            raise RuntimeError("Model creation failed. Check mode and dependencies.")

        if self.verbose:
            model_name = type(self.model).__name__
            print(f"Training {model_name}...")

        # ===== train =====
        # ---------------- TRAIN TEST SPLIT ----------------

        from sklearn.model_selection import train_test_split

        stratify_y = y if self.task == "classification" else None

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=self.random_state,
            stratify=stratify_y,
        )

        # ---------------- TRAIN ----------------

        self.model.fit(X_train, y_train)

# predictions for metrics
        y_pred = self.model.predict(X_test)

# ---------------- METRICS ----------------

        if self.task == "classification":
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average="weighted")

            self.metrics_ = {
                "accuracy": acc,
                "f1": f1,
            }

        else:  # regression
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)

            self.metrics_ = {
                "r2": r2,
                "mae": mae,
            }

        self.trained = True

        # ---------------- AUTO SAVE ----------------
        if self.auto_save_path is not None:
            try:
                self.save(self.auto_save_path, verbose=self.verbose)
            except Exception as e:
                print(f"Auto-save failed: {e}")

# ---------------- VERBOSE LOG ----------------

        if self.verbose:
            primary_metric = (
                self.metrics_["accuracy"]
                if self.task == "classification"
                else self.metrics_["r2"]
            )
            name = "Accuracy" if self.task == "classification" else "R2"

            print(f"Model trained | {name}: {primary_metric:.4f}")
        
    def _prepare_input(self, data):
        """
        Internal helper to preprocess inference data.
        Supports dict, list, and pandas DataFrame.
        """
        import pandas as pd

        # ---------- dict ----------
        if isinstance(data, dict):
            X = pd.DataFrame([data])

        # ---------- list ----------
        elif isinstance(data, list):
            if len(data) == 0:
                raise ValueError("Input data list is empty.")

            # list of dicts
            if isinstance(data[0], dict):
                X = pd.DataFrame(data)
            else:
                # assume list of lists
                X = pd.DataFrame(data, columns=self.columns)

        # ---------- DataFrame ----------
        elif hasattr(data, "columns") and hasattr(data, "iloc"):
            # more robust than isinstance check
            X = data.copy()

        else:
            raise ValueError(
                f"Unsupported input type for prediction: {type(data)}"
            )

        # ---------- Column alignment ----------
        if self.columns is not None:
            missing_cols = set(self.columns) - set(X.columns)
            for col in missing_cols:
                X[col] = 0

            X = X[self.columns]

        # ---------- Apply preprocessing ----------
        if self.use_preprocess and self.preprocessor is not None:
            X = self.preprocessor.transform(X)

        return X
    

    # ================= PREDICT =================
    def predict(self, new_data):
        """
        Predict on new data.

        Supported input formats:
        - pandas DataFrame
        - dict
        - list of dicts
        - list / list of lists
        """

        if not self.trained:
            raise RuntimeError("Model is not trained yet. Call train() first.")

        # 🔥 unified preprocessing pipeline
        X = self._prepare_input(new_data)

        return self.model.predict(X)

    def feature_importance(self):
        """
        Return feature importance as a sorted pandas Series.
        Works for tree-based models like RandomForest and LightGBM.
        """

        import pandas as pd

        if not self.trained:
            raise RuntimeError("Model is not trained yet. Call train() first.")

        if not hasattr(self.model, "feature_importances_"):
            raise RuntimeError(
                "Current model does not support feature importance."
            )

        importance = self.model.feature_importances_

        fi = pd.Series(importance, index=self.columns)
        fi = fi.sort_values(ascending=False)

        return fi
    


    def score(self):
        """
        Return primary metric.
        - accuracy for classification
        - r2 for regression
        """

        if not self.trained:
            raise RuntimeError("Model is not trained yet.")

        if self.metrics_ is None:
            raise RuntimeError("Metrics not available.")

        if self.task == "classification":
            return self.metrics_["accuracy"]
        else:
            return self.metrics_["r2"]


    # ================= SAVE =================
    def save(self, path, verbose=True):
        """
        Save trained AutoModel to disk.
        """

        if not self.trained:
            raise RuntimeError("Cannot save before training.")

        # 🔥 normalize here (CRITICAL FIX)
        from .helpers import _normalize_save_path
        path = _normalize_save_path(path)

        package = {
            "model": self.model,
            "preprocessor": self.preprocessor,
            "columns": self.columns,
            "task": self.task,
            "use_preprocess": self.use_preprocess,
            "scale": self.scale,
            "random_state": self.random_state,
            "trained": self.trained,
            "version": getattr(self, "__version__", "0.1.0"),
        }

        from .persistence import save_object
        save_object(package, path)

        if verbose:
            print(f"Model saved as {path}")
    # ================= LOAD =================
    @classmethod
    def load(cls, path, verbose=True):
        """Load AutoModel from disk."""
        package = load_object(path)

        obj = cls(
            task=package["task"],
            preprocess=package["use_preprocess"],
            scale=package["scale"],
            verbose=verbose,
            random_state=package["random_state"],
        )

        obj.model = package["model"]
        obj.preprocessor = package["preprocessor"]
        obj.columns = package["columns"]
        obj.trained = package["trained"]

        if verbose:
            model_name = type(obj.model).__name__
            feature_count = len(obj.columns)

            print(f"Loaded {model_name} "f"({obj.task}) | Features: {feature_count}")

        return obj
    
    def predict_proba(self, data):
        """
        Predict class probabilities (classification only).

        Returns
        -------
        List[Dict[class_label, probability]]
        """

        if not self.trained:
            raise RuntimeError("Model is not trained yet.")

        if self.task != "classification":
            raise RuntimeError(
                "predict_proba is only available for classification tasks."
            )

        if not hasattr(self.model, "predict_proba"):
            raise RuntimeError(
                "Underlying model does not support probability prediction."
            )

        # 🔄 Use SAME preprocessing pipeline as predict()
        X = self._prepare_input(data)

        probs = self.model.predict_proba(X)
        classes = self.model.classes_

        # 🏷️ Convert to labeled probabilities
        labeled_probs = []
        for row in probs:
            labeled_row = {
                (cls.item() if hasattr(cls, "item") else cls): float(prob)
                for cls, prob in zip(classes, row)
            }
            labeled_probs.append(labeled_row)

        return labeled_probs