from pathlib import Path

import joblib


def save_pipeline(
    pipeline,
    path: Path,
) -> None:
    """
    Persist a fitted ML pipeline to disk.
    """

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        pipeline,
        path,
    )


def load_pipeline(path: Path):
    """
    Load a persisted ML pipeline from disk.
    """

    return joblib.load(path)