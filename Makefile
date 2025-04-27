.PHONY: tool-check
tool-check:
	pip install -U ruff

.PHONY: format
format:
	ruff format
	ruff check --fix
	ruff format

.PHONY: test
test:
	pytest -rP

.PHONY: all
all:
	make format
	make test
