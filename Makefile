.PHONY: install dev test lint fmt serve check ingest reset clean publish

install:
	uv sync

dev:
	uv sync --extra dev

test:
	uv run pytest -q

lint:
	uv run ruff check .

fmt:
	uv run ruff format .

serve:
	uv run ragforge serve --reload

check:
	uv run ragforge check

ingest:
	uv run ragforge ingest samples

reset:
	uv run ragforge reset --yes

clean:
	rm -rf data .pytest_cache .ruff_cache

publish:
	scripts/publish.sh
