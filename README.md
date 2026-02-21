
# pyezml 🚀  
**Beginner-Friendly AutoML for Tabular Data**

![PyPI version](https://img.shields.io/pypi/v/pyezml)
![Python](https://img.shields.io/pypi/pyversions/pyezml)
![License](https://img.shields.io/github/license/ajayy51/pyezml)
![Downloads](https://img.shields.io/pypi/dm/pyezml)
![GitHub stars](https://img.shields.io/github/stars/ajayy51/pyezml?style=social)

Train machine learning models in **one line of code** — no ML expertise required.

pyezml is a lightweight yet powerful AutoML library that automatically handles preprocessing, model selection, and prediction so you can focus on results.

Built for students, developers, analysts, and beginners who want fast, reliable predictions without complex pipelines.

---

## 🚀 What's New in v0.2.0

-  **Labeled probability predictions**
-  **Auto-save via `save=` parameter**
-  **Automatic `.pkl` extension handling**
-  **Robust DataFrame prediction support**
-  **Built-in sample data generators**
-  **Unified prediction pipeline**

---

## 🚀 Installation

```bash
pip install pyezml
````

**Optional (recommended for best mode):**

```bash
pip install lightgbm
```

**Requirements**

* Python ≥ 3.8

---

## ⚡ Quick Example

```python
from ezml import train_model

model = train_model("data.csv", target="price")
print(model.score())

```

That’s it — model trained and evaluated.

---

## 🧪 Generate Sample Data (NEW)

No dataset? No problem.

```python
from ezml.datasets import make_classification_data
from ezml import train_model

df = make_classification_data()

model = train_model(df, target="target")
print(model.score())
```

Perfect for quick testing and demos.

---


## 🧬 Ultimate Synthetic Data Generation

Generate advanced datasets with probabilistic distributions and mathematical feature engineering:

```python
from ezml.datasets import make_mathematical_synthetic_data, list_supported_distributions

print(list_supported_distributions())

df = make_mathematical_synthetic_data(
    n_samples=2000,
    task="classification",
    target_name="label"
)

print(df.head())
```

You can also build custom schemas with normal, uniform, gamma, beta, poisson, binomial, triangular and more.

## 🔮 Labeled Probability Predictions (NEW)

pyezml returns **human-readable probabilities**:

```python
probs = model.predict_proba({
    "feature_0": 0.5,
    "feature_1": -1.2
})

print(probs)
```

Example output:

```python
[{'No': 0.12, 'Yes': 0.88}]
```

No index guessing required.

---

## 💾 Auto-Save Models (NEW)

Save automatically during training:

```python
model = train_model(
    df,
    target="target",
    save="my_model"  # .pkl added automatically
)
```

Manual save still works:

```python
model.save("model.pkl")
```

---

## 🔧 Advanced Usage

```python
from ezml import AutoModel

model = AutoModel(mode="best")  # fast | best
model.train("data.csv", target="price")

print(model.score())
print(model.feature_importance())
```

---

## ⚡ Model Modes

pyezml provides two performance modes:

### 🚀 fast (default)

* **Model:** RandomForest
* **Best for:** small to medium datasets
* **Why use it:** fast, robust, beginner-safe

### 🔥 best

* **Model:** LightGBM
* **Best for:** larger datasets and higher accuracy
* **Why use it:** stronger learning on complex tabular data

> 💡 Automatically falls back to RandomForest if LightGBM is unavailable.

---

## 📊 Metrics API

After training:

### Classification

* Accuracy
* F1-score

### Regression

* R² score
* MAE

Example:

```python
print(model.metrics_)
print(model.score())  # primary metric
```

---

## 🔮 Flexible Prediction Inputs

### Dict (recommended)

```python
model.predict({"feature1": value1, "feature2": value2})
```

### Batch dict

```python
model.predict([
    {"feature1": v1, "feature2": v2},
    {"feature1": v3, "feature2": v4}
])
```

### pandas DataFrame

```python
model.predict(df)
```

---

## 🧹 Automatic Preprocessing

pyezml automatically handles:

* Missing value imputation
* Categorical encoding
* Optional feature scaling
* Column alignment during prediction

No manual preprocessing required.

---

## 📓 Demo Notebook

See the full working example:

👉 examples/pyezml_demo.ipynb

---

## 🎯 Project Goal

pyezml aims to make machine learning:

* simple
* fast
* accessible
* beginner-friendly

without sacrificing real-world usability.

---

## 🤝 Contributing

Contributions, issues, and suggestions are welcome!

If you find a bug or have an idea:

1. Fork the repo
2. Create a feature branch
3. Submit a pull request

---

## 📜 License

MIT License — free to use and modify.

---

## ⭐ Support

If you find pyezml useful, consider giving the repository a star ⭐
It helps the project grow!


