import pathlib


PATH_TO_TESTS: pathlib.Path = pathlib.Path(__file__).resolve().parent
PATH_TO_TMP: pathlib.Path = PATH_TO_TESTS / "to_delete"
