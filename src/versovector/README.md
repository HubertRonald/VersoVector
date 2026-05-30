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

```bash
python -m src.versovector.training.build_dataset --config configs/model_config.toml
python -m src.versovector.training.train_features --config configs/model_config.toml
python -m src.versovector.training.train_supervised --config configs/model_config.toml
python -m src.versovector.training.train_unsupervised --config configs/model_config.toml
python -m src.versovector.training.register_model --config configs/model_config.toml
```

next step:

```bash
python -m src.versovector.training.pipeline --config configs/model_config.toml
```