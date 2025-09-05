.PHONY: help install-python install-julia setup-env test clean

help:  ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install-python:  ## Install Python dependencies
	pip install -e .
	pip install -e .[dev]

install-julia:  ## Install Julia dependencies
	julia --project=. -e "using Pkg; Pkg.add([\"JuMP\", \"Ipopt\", \"OSQP\", \"CSV\", \"DataFrames\", \"JSON\", \"ArgParse\", \"Logging\"])"
	julia --project=. -e "using Pkg; Pkg.add(\"MixedComplementarityProblems\")"

setup-env: install-python install-julia  ## Set up both Python and Julia environments
	@echo "Environment setup complete!"

test:  ## Run tests
	python -m pytest tests/ -v

test-julia:  ## Test Julia MCP solver
	julia --project=. -e "using MixedComplementarityProblems; println(\"MixedComplementarityProblems.jl loaded successfully!\")"

clean:  ## Clean up build artifacts
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
