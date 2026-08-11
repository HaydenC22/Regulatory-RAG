.PHONY: up down ingest eval test lint fmt

up:
	docker compose up --build

down:
	docker compose down

ingest:
	docker compose exec api python -m services.ingestion.load

eval:
	# --config final also re-runs baseline internally to produce the comparison report + chart
	docker compose exec api python -m eval.run_eval --config final

test:
	pytest tests/unit -v
	pytest tests/integration -v

lint:
	ruff check .
	black --check .

fmt:
	ruff check --fix .
	black .
