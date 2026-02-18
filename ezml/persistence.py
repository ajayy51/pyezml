import joblib
import os

def save_object(obj, path):
    """Save object to disk."""
    joblib.dump(obj, path)


def load_object(path):
    """Load object from disk."""
    return joblib.load(path)

def save_model(model, filename):
    """
    Save trained model to disk.
    """

    if not filename.endswith(".pkl"):
        filename = filename + ".pkl"

    os.makedirs("models", exist_ok=True)

    path = os.path.join("models", filename)
    joblib.dump(model, path)

    print(f"Model saved to {path}")