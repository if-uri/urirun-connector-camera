.PHONY: help test
help: ## List targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  %-9s %s\n",$$1,$$2}'
test: ## Install editable + pytest (pytest pinned here: most connectors don't declare a [test] extra)
	pip install -e . pytest
	python3 -m pytest -q
