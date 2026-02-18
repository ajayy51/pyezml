import os


def _normalize_save_path(path: str):
    """Ensure model path has .pkl extension and handle empty values."""
    if path is None:
        return None

    path = path.strip()

    # 🚨 handle empty string
    if path == "":
        return None

    # If user already provided extension → keep it
    if os.path.splitext(path)[1]:
        return path

    return path + ".pkl"


def train_model(
    data,
    target,
    task="auto",
    mode="fast",
    preprocess=True,
    scale=False,
    verbose=True,
    random_state=42,
    save=None,
):
    """
    One-line training helper for ezml.
    """

    # 🔥 IMPORT HERE (lazy import — breaks circular dependency)
    from .automodel import AutoModel

    # normalize save path
    save = _normalize_save_path(save)

    model = AutoModel(
        task=task,
        mode=mode,
        preprocess=preprocess,
        scale=scale,
        verbose=verbose,
        random_state=random_state,
        save=save,
    )

    model.train(data, target=target)
    return model