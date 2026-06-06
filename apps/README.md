# Apps

This directory contains user-facing applications built on top of VersoVector services.

Current apps:

```text
apps/
└── frontend/
```

## Purpose

The apps layer contains application code.

It is different from:

```text
src/versovector/
    Python package code.

services/
    Docker and service packaging.

tests/
    Unit and integration tests.
```

## Current Frontend

The first frontend is a Gradio demo for emotional-semantic poem analysis.

It calls the FastAPI service instead of loading model artifacts directly.