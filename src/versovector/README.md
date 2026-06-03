# Convert notebooks to real scripts

```text
build_dataset.py
    equivalent to notebook 01

train_features.py
    equivalent to notebook 02

train_supervised.py
    equivalent to notebook 03

train_unsupervised.py
    equivalent to notebook 04

register_model.py
    MLflow tracking + model bundle registration
```

expected commands

check SyntaxError

```bash
python -m py_compile \
  src/versovector/training/mlflow_utils.py \
  src/versovector/training/build_dataset.py \
  src/versovector/training/train_features.py \
  src/versovector/training/train_supervised.py \
  src/versovector/training/train_unsupervised.py \
  src/versovector/training/register_model.py
```


excecute order (search package src and .)

```bash
PYTHONPATH=src:. python -m versovector.training.build_dataset --config configs/model_config.toml && \
PYTHONPATH=src:. python -m versovector.training.train_features --config configs/model_config.toml && \
PYTHONPATH=src:. python -m versovector.training.train_supervised --config configs/model_config.toml && \
PYTHONPATH=src:. python -m versovector.training.train_unsupervised --config configs/model_config.toml && \
PYTHONPATH=src:. python -m versovector.training.register_model --config configs/model_config.toml
```
