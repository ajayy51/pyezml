````markdown
# pyezml 🚀  
**Beginner-Friendly AutoML for Tabular Data**

![PyPI version](https://img.shields.io/pypi/v/pyezml)
![Python](https://img.shields.io/pypi/pyversions/pyezml)
![License](https://img.shields.io/github/license/ajayy51/pyezml)
![Downloads](https://img.shields.io/pypi/dm/pyezml)

pyezml is a lightweight AutoML library that lets you train powerful machine learning models in just a few lines of code — no deep ML knowledge required.

Built for students, developers, analysts, and beginners who want fast, reliable predictions without complex pipelines.

---

## ✨ Key Features

- 🧠 Smart task detection (classification vs regression)  
- ⚡ Fast vs Best model modes  
- 🧹 Automatic preprocessing (missing values + encoding)  
- 📊 Built-in metrics API  
- 🔮 Safe dict-based prediction  
- 💾 Built-in save & load  
- 🐼 Supports CSV and pandas DataFrame  
- 🪶 Lightweight and beginner-friendly  

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

* Python >= 3.8

---

## ⚡ Quick Example

```python
from ezml import train_model

model = train_model("data.csv", target="price")
print(model.predict({"area_sqft": 3000, "bedrooms": 3}))
```

That’s it — model trained and ready.

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

pyezml provides two performance modes to balance speed and accuracy.

---

**🚀 fast (default)**

* **Model:** RandomForest
* **Best for:** small to medium datasets
* **Why use it:** fast, robust, beginner-safe

---

**🔥 best**

* **Model:** LightGBM
* **Best for:** larger datasets and higher accuracy
* **Why use it:** more powerful learning on complex tabular data

> 💡 Automatically falls back to RandomForest if LightGBM is unavailable or dataset is very small.

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

### List

```python
model.predict([[v1, v2, v3]])
```

### Batch dict

```python
model.predict([
    {"feature1": v1, "feature2": v2},
    {"feature1": v3, "feature2": v4}
])
```

---

## 💾 Save and Load Models

### Save

```python
model.save("model.pkl")
```

### Load

```python
from ezml import AutoModel

loaded = AutoModel.load("model.pkl")
print(loaded.predict({...}))
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

```
::contentReference[oaicite:0]{index=0}
```
