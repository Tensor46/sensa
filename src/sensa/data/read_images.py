import logging
import os
import pathlib

from sensa.assets import TESTRUN


def read_label_per_folder(
    path: pathlib.Path,
    start_label_id_at: int = 0,
    stop_label_id_at: int | None = None,
    max_samples_per_label: float | int | None = None,
    dbase: list[tuple[pathlib.Path, int]] | None = None,
    extensions: tuple[str, ...] = (".jpeg", ".jpg", ".png", ".tiff", ".webp"),
) -> list[tuple[pathlib.Path, int]]:
    """Scan subfolders of `path`, assign sequential label IDs, and collect image file paths.

    Args:
        path (pathlib.Path):
            Root directory where each immediate subdirectory represents a class label.
        start_label_id_at (int, default=0):
            Integer to start labeling samples from.
        stop_label_id_at (int | None, default=None):
            Exclusive upper bound on label IDs; if reached, remaining folders are skipped.
        max_samples_per_label (int | float | None, default=None):
            If int, maximum files per label; if float, fraction of files per label.
        dbase (list[tuple[pathlib.Path, int]] | None, default=None):
            Existing list to extend, otherwise a new list is created.
        extensions (tuple[str, ...], default=(".jpeg", ".jpg", ".png", ".tiff", ".webp")):
            Allowed file extensions, case-insensitive.

    Returns:
        list[tuple[pathlib.Path, int]]:
        A list of `(file_path, label)` pairs for all images found.
    """
    if not isinstance(path, pathlib.Path):
        logging.error("read_folder_per_label: path must be pathlib.Path")
        raise TypeError("read_folder_per_label: path must be pathlib.Path")
    if not path.is_dir():
        logging.error("read_folder_per_label: not a valid path")
        raise ValueError("read_folder_per_label: not a valid path")

    # find all folders with images
    # assumes any folder with images as a unique label
    ddict: dict[str, list[pathlib.Path]] = {}
    for dpath, ds, file_names in os.walk(path):
        level = dpath[len(str(path)) :].count(os.sep)
        if level >= 2:
            ds[:] = []
        if level != 1:
            continue

        for file_name in file_names[:250] if TESTRUN else file_names:
            if not file_name.lower().endswith(extensions):
                continue
            if dpath not in ddict:
                ddict[dpath] = []
            ddict[dpath].append(file_name)
        if TESTRUN and len(ddict) >= 100:
            break

    # build or extend database of (file_path, label_id)
    dbase: list[tuple[pathlib.Path, int]] = dbase or []
    for i, label in enumerate(sorted(ddict.keys())):
        label_id: int = i + start_label_id_at
        if stop_label_id_at is not None and label_id >= stop_label_id_at:
            break  # reached max labels

        for j, file_name in enumerate(ddict[label]):
            if isinstance(max_samples_per_label, float) and j >= max(
                1, round(max_samples_per_label * len(ddict[label]))
            ):
                break  # reached max samples per label
            if isinstance(max_samples_per_label, int) and j >= max_samples_per_label:
                break  # reached max samples per label

            dbase.append((pathlib.Path(label) / file_name, label_id))
    return dbase
