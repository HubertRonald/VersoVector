locals {
  api_service_name      = "${var.resource_prefix}-api"
  frontend_service_name = "${var.resource_prefix}-frontend"

  api_service_account_id = substr(replace("${var.resource_prefix}-api-sa", "_", "-"), 0, 30)
  web_service_account_id = substr(replace("${var.resource_prefix}-web-sa", "_", "-"), 0, 30)

  required_services = [
    "artifactregistry.googleapis.com",
    "run.googleapis.com",
    "iam.googleapis.com",
  ]
}

resource "google_project_service" "required_apis" {
  for_each = var.enable_project_services ? toset(local.required_services) : toset([])

  project            = var.project_id
  service            = each.key
  disable_on_destroy = false
}

resource "google_artifact_registry_repository" "services" {
  project       = var.project_id
  location      = var.region
  repository_id = var.artifact_registry_repository_id
  description   = "Docker images for the public VersoVector Cloud Run blueprint."
  format        = "DOCKER"
  labels        = var.labels

  depends_on = [google_project_service.required_apis]
}

resource "google_service_account" "api_runtime" {
  project      = var.project_id
  account_id   = local.api_service_account_id
  display_name = "VersoVector API runtime service account"
  description  = "Sanitized runtime identity for the VersoVector API Cloud Run blueprint."

  depends_on = [google_project_service.required_apis]
}

resource "google_service_account" "frontend_runtime" {
  project      = var.project_id
  account_id   = local.web_service_account_id
  display_name = "VersoVector frontend runtime service account"
  description  = "Sanitized runtime identity for the VersoVector frontend Cloud Run blueprint."

  depends_on = [google_project_service.required_apis]
}

resource "google_cloud_run_v2_service" "api" {
  project             = var.project_id
  name                = local.api_service_name
  location            = var.region
  ingress             = "INGRESS_TRAFFIC_ALL"
  deletion_protection = false
  labels              = var.labels

  template {
    service_account = google_service_account.api_runtime.email

    scaling {
      min_instance_count = var.api_min_instances
      max_instance_count = var.api_max_instances
    }

    containers {
      image = var.api_image

      ports {
        container_port = var.api_container_port
      }

      resources {
        limits = {
          cpu    = var.api_cpu
          memory = var.api_memory
        }
        startup_cpu_boost = true
      }

      env {
        name  = "PORT"
        value = tostring(var.api_container_port)
      }

      env {
        name  = "VERSOVECTOR_MODEL_BUNDLE_DIR"
        value = "/app/artifacts/model_bundle"
      }

      env {
        name  = "VERSOVECTOR_MODEL_BUNDLE_URI"
        value = var.model_bundle_uri
      }
    }
  }

  traffic {
    type    = "TRAFFIC_TARGET_ALLOCATION_TYPE_LATEST"
    percent = 100
  }

  depends_on = [
    google_artifact_registry_repository.services,
    google_service_account.api_runtime,
  ]
}

resource "google_cloud_run_v2_service" "frontend" {
  project             = var.project_id
  name                = local.frontend_service_name
  location            = var.region
  ingress             = "INGRESS_TRAFFIC_ALL"
  deletion_protection = false
  labels              = var.labels

  template {
    service_account = google_service_account.frontend_runtime.email

    scaling {
      min_instance_count = var.frontend_min_instances
      max_instance_count = var.frontend_max_instances
    }

    containers {
      image = var.frontend_image

      ports {
        container_port = var.frontend_container_port
      }

      resources {
        limits = {
          cpu    = var.frontend_cpu
          memory = var.frontend_memory
        }
        startup_cpu_boost = true
      }

      env {
        name  = "PORT"
        value = tostring(var.frontend_container_port)
      }

      env {
        name  = "VERSOVECTOR_API_BASE_URL"
        value = google_cloud_run_v2_service.api.uri
      }

      env {
        name  = "VERSOVECTOR_API_TIMEOUT_SECONDS"
        value = tostring(var.api_timeout_seconds)
      }
    }
  }

  traffic {
    type    = "TRAFFIC_TARGET_ALLOCATION_TYPE_LATEST"
    percent = 100
  }

  depends_on = [
    google_artifact_registry_repository.services,
    google_cloud_run_v2_service.api,
    google_service_account.frontend_runtime,
  ]
}

resource "google_cloud_run_v2_service_iam_member" "frontend_public_invoker" {
  count = var.allow_public_frontend_access ? 1 : 0

  project  = var.project_id
  location = var.region
  name     = google_cloud_run_v2_service.frontend.name
  role     = "roles/run.invoker"
  member   = "allUsers"
}

resource "google_cloud_run_v2_service_iam_member" "api_public_invoker" {
  count = var.allow_public_api_access ? 1 : 0

  project  = var.project_id
  location = var.region
  name     = google_cloud_run_v2_service.api.name
  role     = "roles/run.invoker"
  member   = "allUsers"
}

resource "google_cloud_run_v2_service_iam_member" "api_frontend_invoker" {
  count = var.allow_public_api_access ? 0 : 1

  project  = var.project_id
  location = var.region
  name     = google_cloud_run_v2_service.api.name
  role     = "roles/run.invoker"
  member   = "serviceAccount:${google_service_account.frontend_runtime.email}"
}
