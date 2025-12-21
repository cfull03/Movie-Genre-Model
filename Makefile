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

## Install pre-commit hooks
.PHONY: pre-commit-install
pre-commit-install:
	pre-commit install
	@echo ">>> Pre-commit hooks installed successfully"
	@echo ">>> Run 'pre-commit run --all-files' to test on all files"



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


## Preprocess data (WARNING: Fits on entire dataset - for exploration/evaluation only)
.PHONY: preprocess
preprocess: data
	$(PYTHON_INTERPRETER) descriptions/modeling/preprocess.py


## Train model (splits data before preprocessing to prevent data leakage)
.PHONY: train
train: data
	$(PYTHON_INTERPRETER) descriptions/modeling/train.py

## Evaluate model (optionally specify MODEL_PATH, e.g., make evaluate MODEL_PATH=logisticregression.joblib)
## Note: Uses interim data by default. Use --use-processed flag to evaluate on processed data.
.PHONY: evaluate
evaluate: train
ifdef MODEL_PATH
	$(PYTHON_INTERPRETER) descriptions/modeling/evaluate.py --model-path $(MODEL_PATH)
else
	$(PYTHON_INTERPRETER) descriptions/modeling/evaluate.py
endif


.PHONY: plots
plots:
	$(PYTHON_INTERPRETER) descriptions/plots.py


## Predict genres from movie descriptions
## Usage examples:
##   make predict DESCRIPTION="A thrilling action movie"
##   make predict INPUT_FILE=data/test_movies.csv
##   make predict INPUT_FILE=data/test_movies.csv OUTPUT_FILE=predictions.csv MODEL_PATH=linearsvc.joblib THRESHOLD=0.5
.PHONY: predict
predict: train
ifdef DESCRIPTION
	$(PYTHON_INTERPRETER) descriptions/modeling/predict.py --description "$(DESCRIPTION)" \
		$(if $(MODEL_PATH),--model-path $(MODEL_PATH)) \
		$(if $(THRESHOLD),--threshold $(THRESHOLD))
else ifdef INPUT_FILE
	$(PYTHON_INTERPRETER) descriptions/modeling/predict.py --input-file $(INPUT_FILE) \
		$(if $(MODEL_PATH),--model-path $(MODEL_PATH)) \
		$(if $(OUTPUT_FILE),--output-file $(OUTPUT_FILE)) \
		$(if $(DESCRIPTION_COLUMN),--description-column $(DESCRIPTION_COLUMN)) \
		$(if $(THRESHOLD),--threshold $(THRESHOLD))
else
	@echo "Error: Please provide either DESCRIPTION or INPUT_FILE"
	@echo "Examples:"
	@echo "  make predict DESCRIPTION=\"A thrilling action movie\""
	@echo "  make predict INPUT_FILE=data/test_movies.csv"
	@echo "  make predict INPUT_FILE=data/test_movies.csv OUTPUT_FILE=predictions.csv"
	@exit 1
endif

## Start FastAPI server (development)
.PHONY: api
api:
	$(PYTHON_INTERPRETER) app/run.py

## Start FastAPI server with uvicorn (production)
.PHONY: api-prod
api-prod:
	uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4

## Start Flask API server (legacy)
.PHONY: api-flask
api-flask:
	$(PYTHON_INTERPRETER) app/run.py

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
