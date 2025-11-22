#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = movie_genre_model
PYTHON_VERSION = 3.12
PYTHON_INTERPRETER = python

#################################################################################
# COMMANDS                                                                      #
#################################################################################


## Install Python dependencies
.PHONY: requirements
requirements:
	conda env update --name $(PROJECT_NAME) --file environment.yml --prune
	



## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

.PHONY: reset
reset: clean
	find . -type f -name "*.joblib" -delete
	find . -type d -name "models" -delete
	find . -type d -name "data" -delete
	find . -type d -name "reports" -delete
	find . -type d -name "figures" -delete
	find . -type d -name "notebooks" -delete
	find . -type d -name "tests" -delete
	find . -type d -name "docs" -delete
	find . -type d -name "examples" -delete


## Lint using ruff (use `make format` to do formatting)
.PHONY: lint
lint:
	ruff format --check
	ruff check

## Format source code with ruff
.PHONY: format
format:
	ruff check --fix
	ruff format



## Run tests
.PHONY: test
test:
	python -m pytest tests


## Set up Python interpreter environment
.PHONY: create_environment
create_environment:
	conda env create --name $(PROJECT_NAME) -f environment.yml
	
	@echo ">>> conda env created. Activate with:\nconda activate $(PROJECT_NAME)"
	



#################################################################################
# PROJECT RULES                                                                 #
#################################################################################


## Make dataset
.PHONY: data
data: requirements
	$(PYTHON_INTERPRETER) descriptions/dataset.py


.PHONY: preprocess
preprocess: data
	$(PYTHON_INTERPRETER) descriptions/modeling/preprocess.py


.PHONY: train
train: preprocess
	$(PYTHON_INTERPRETER) descriptions/modeling/train.py

## Evaluate model (optionally specify MODEL_PATH, e.g., make evaluate MODEL_PATH=logisticregression.joblib)
.PHONY: evaluate
evaluate: train
ifdef MODEL_PATH
	$(PYTHON_INTERPRETER) descriptions/modeling/evaluate.py --model-path $(MODEL_PATH)
else
	$(PYTHON_INTERPRETER) descriptions/modeling/evaluate.py
endif

#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
