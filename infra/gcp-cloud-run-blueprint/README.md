# GCP Cloud Run Blueprint

This directory contains a sanitized Terraform blueprint for deploying the VersoVector public portfolio architecture to Google Cloud Run.

It is intentionally not a production deployment.

```text
infra/
└── gcp-cloud-run-blueprint/
    ├── README.md
    ├── main.tf
    ├── variables.tf
    ├── outputs.tf
    ├── terraform.tfvars.example
    └── versions.tf
```

## Purpose

This blueprint documents how the public VersoVector architecture could be deployed without exposing real production infrastructure.

It demonstrates:

- an Artifact Registry Docker repository;
- a Cloud Run API service;
- a Cloud Run frontend service;
- separate runtime service accounts;
- conceptual API and frontend IAM boundaries;
- generic variables and outputs;
- a safe public/private repository separation.

## Public vs Private Boundary

The public repository may include:

- architecture diagrams;
- local services;
- Docker packaging;
- sanitized Terraform blueprints;
- generic variables;
- non-production examples.

A private repository should contain:

- real GCP project IDs;
- Terraform remote state backend configuration;
- production `terraform.tfvars` files;
- real IAM bindings;
- secrets;
- private Cloud Run settings;
- authentication and billing logic;
- premium feature gating;
- private datasets and model artifact locations;
- production monitoring and analytics.

## Architecture

```text
Browser
  |
  v
Cloud Run Frontend
  |
  | HTTP request to API service
  v
Cloud Run API
  |
  | Future remote model bundle retrieval
  v
Model Artifact Store
  - GCS
  - MLflow artifact store
  - other controlled artifact location

Artifact Registry
  - versovector-api image
  - versovector-frontend image
```

## Resources Created

This blueprint defines:

- `google_artifact_registry_repository.services`
- `google_service_account.api_runtime`
- `google_service_account.frontend_runtime`
- `google_cloud_run_v2_service.api`
- `google_cloud_run_v2_service.frontend`
- optional public invoker binding for the frontend
- optional public invoker binding for the API
- private API invoker binding for the frontend runtime service account

## Important Notes

### Container images must already exist

This blueprint does not build or push Docker images.

Build and push images before running `terraform apply`.

Example image format:

```text
REGION-docker.pkg.dev/PROJECT_ID/REPOSITORY/versovector-api:TAG
REGION-docker.pkg.dev/PROJECT_ID/REPOSITORY/versovector-frontend:TAG
```

### Model artifacts are not included

The public Docker image should package application code, not generated model artifacts.

Generated model artifacts such as `.joblib` files, local MLflow runs, and `artifacts/model_bundle/` should not be committed to Git.

This blueprint includes `model_bundle_uri` as a placeholder for a future controlled artifact location.

### Private API behavior

By default:

```text
allow_public_frontend_access = true
allow_public_api_access      = false
```

This expresses the target architecture:

```text
Browser -> Frontend Cloud Run -> Private API Cloud Run
```

However, the current public Gradio frontend does not yet implement authenticated service-to-service ID token calls.

For local or temporary public demo experiments, either:

1. set `allow_public_api_access = true`, or
2. extend the frontend API client to call private Cloud Run services with identity-aware authentication.

Production authentication details belong in a private deployment repository.

## Usage

From the repository root:

```bash
cd infra/gcp-cloud-run-blueprint
cp terraform.tfvars.example terraform.tfvars
```

Edit `terraform.tfvars` with sandbox values only.

Then run:

```bash
terraform init
terraform fmt -recursive
terraform validate
terraform plan -var-file=terraform.tfvars
```

Apply only in a controlled sandbox project:

```bash
terraform apply -var-file=terraform.tfvars
```

## What This Blueprint Does Not Include

This public blueprint intentionally excludes:

- production project IDs;
- production remote state backend;
- private domains;
- secret manager values;
- service account keys;
- billing/subscription logic;
- real customer data;
- production observability;
- private dataset/model artifact paths;
- environment-specific dev/staging/prod folders.

## Repository Strategy

```text
Public repository:
    This is how the architecture could be deployed.

Private repository:
    This is the actual deployment.
```

A future private repository such as `VersoVector-Platform` should own production Terraform, IAM, secrets, billing, auth, premium features, analytics, and real deployment configuration.
