# Antangle

Fruit quality prediction pipeline for the Antangle project.

## Project Overview

This repository contains a machine learning pipeline for predicting fruit quality from sensor or feature data. The project includes:

- `data_exploration.py`: Exploratory data analysis and dataset understanding
- `preprocessing.py`: Data cleaning, encoding, and preprocessing
- `model_training.py`: Training baseline machine learning models
- `optimized_models.py`: Hyperparameter tuning and optimized model selection
- `visualization.py`: Plot generation and evaluation visualization
- `main.py`: Main script to run the full pipeline
- `requirements.txt`: Project dependencies

## Usage

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Run the full pipeline:

```bash
python main.py
```

3. Inspect outputs in the `outputs/` directory.

## Notes

- Generated artifacts such as plot images and serialized models should be ignored by Git and are handled in `.gitignore`.
- Keep raw data and model outputs out of version control unless explicitly needed.

## Repository

This project is linked to: `https://github.com/eslavathnandini/Antangle.git`
