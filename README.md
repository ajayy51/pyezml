
````markdown
# ezml 🚀  
**Beginner-Friendly AutoML for Tabular Data**

ezml is a lightweight, easy-to-use AutoML library that lets anyone train machine learning models in just a few lines of code — no deep ML knowledge required.

Built for students, developers, analysts, and beginners who want fast, reliable predictions without complex pipelines.

---

## ✨ Key Features

- Smart task detection (classification vs regression)  
- Fast vs Best model modes  
- Automatic preprocessing (missing values + encoding)  
- One-line training helper  
- Dict-based safe prediction  
- Built-in save & load support  
- Lightweight and beginner-friendly  
- Modular and extensible design  

---

## Installation

Clone the repository and install dependencies:

```bash
pip install -r requirements.txt
````

**Optional (recommended for best mode):**

```bash
pip install lightgbm
```

---

## ⚡ Quick Start (Recommended)

### One-line training

```python
from ezml import train_model

model = train_model("data.csv", target="price")

prediction = model.predict([[3000, 3, 2, 5]])
```

---

## 🔧 Advanced Usage

### Using AutoModel directly

```python
from ezml import AutoModel

model = AutoModel(mode="best")  # fast | best
model.train("data.csv", target="price")

preds = model.predict([[3000, 3, 2, 5]])
```

---

## ⚡ Model Modes



ezml provides two performance modes to balance speed and accuracy.

#### 🚀 fast (default)

Model: RandomForest

Best for: small to medium datasets, quick and stable baseline

Why use it: fast training, very robust, beginner-safe

#### 🔥 best

Model: LightGBM

Best for: larger datasets and higher accuracy needs

Why use it: more powerful learning, better performance on complex tabular data

💡 ezml automatically falls back to RandomForest if LightGBM is not installed.

> ezml automatically falls back to RandomForest if LightGBM is not installed.

---

## 🧠 Supported Tasks

ezml automatically detects:

* ✅ Regression problems
* ✅ Binary classification
* ✅ Multi-class classification

Manual override is also supported:

```python
AutoModel(task="classification")
AutoModel(task="regression")
```

---

## 🔮 Flexible Prediction Inputs

ezml accepts multiple input formats:

### Dict (recommended)

```python
model.predict({
    "feature1": value1,
    "feature2": value2
})
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

loaded_model = AutoModel.load("model.pkl")
preds = loaded_model.predict({...})
```

---

## 🧹 Automatic Preprocessing

ezml automatically handles:

* Missing value imputation
* Categorical encoding
* Optional feature scaling
* Column alignment during prediction

No manual preprocessing required.

---

## 🎯 Project Goal

ezml aims to make machine learning:

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



## ⭐ Support

If you find ezml useful, consider giving the repo a star ⭐

It helps the project grow!

```

