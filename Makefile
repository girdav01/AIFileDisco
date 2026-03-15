.PHONY: help install dev scan server test clean testdata docker

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install: ## Install system-wide (pip install .)
	pip install .

dev: ## Install in dev/editable mode
	pip install -e ".[dev]"

scan: ## Quick scan of current directory
	python3 aifiles.py .

server: ## Start the web dashboard on port 8505
	python3 server.py

test: ## Run all tests
	python3 -m pytest tests/ -v 2>/dev/null || python3 -m unittest discover -s tests -v

testdata: ## Generate sample test data
	python3 generate_test_data.py

clean: ## Remove build artifacts and caches
	rm -rf build/ dist/ *.egg-info __pycache__ tests/__pycache__ report/
	find . -name "*.pyc" -delete

docker: ## Build Docker image
	docker build -t aifilefinder .

docker-run: ## Run in Docker (mount current dir as /data)
	docker run --rm -p 8505:8505 -v "$$(pwd):/data" aifilefinder
