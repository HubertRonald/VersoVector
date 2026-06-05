# Tests

This directory contains the test layer for VersoVector.

The test suite is separated into unit tests and integration tests.

```text
tests/
├── unit/
└── integration/
```

## Unit Tests

Unit tests validate isolated Python components without requiring a full model bundle.

Examples:

* text cleaning and tag parsing;
* artifact path helpers;
* schema serialization;
* similarity result formatting;
* tag ranking logic.

Run:

```bash
PYTHONPATH=src:. pytest tests/unit
```

## Integration Tests

Integration tests validate that multiple layers work together.

Examples:

* loading a minimal test model bundle;
* running `PoemAnalyzer.analyze_dict`;
* calling FastAPI endpoints with `TestClient`;
* validating API response shapes.

Run:

```bash
PYTHONPATH=src:. pytest tests/integration
```

## Coverage

Run all tests with coverage:

```bash
PYTHONPATH=src:. pytest --cov=src --cov=modules tests/
```

## Services vs Integration Tests

The `services/` directory is for deployment packaging and service-level assets.

The `tests/integration/` directory is for validating interactions between components.

They are related, but not the same thing:

```text
services/
    How the API or frontend is packaged and deployed.

tests/integration/
    Whether components work together correctly.
```

## Notes

The integration tests use a minimal mocked model bundle generated under `tmp_path`.

They do not require the real `artifacts/model_bundle/` directory and do not use large model artifacts.
