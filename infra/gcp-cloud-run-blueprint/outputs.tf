output "artifact_registry_repository" {
  description = "Artifact Registry Docker repository name."
  value       = google_artifact_registry_repository.services.name
}

output "api_service_name" {
  description = "Cloud Run API service name."
  value       = google_cloud_run_v2_service.api.name
}

output "api_service_uri" {
  description = "Cloud Run API service URI."
  value       = google_cloud_run_v2_service.api.uri
}

output "frontend_service_name" {
  description = "Cloud Run frontend service name."
  value       = google_cloud_run_v2_service.frontend.name
}

output "frontend_service_uri" {
  description = "Cloud Run frontend service URI."
  value       = google_cloud_run_v2_service.frontend.uri
}

output "api_runtime_service_account" {
  description = "Runtime service account used by the API service."
  value       = google_service_account.api_runtime.email
}

output "frontend_runtime_service_account" {
  description = "Runtime service account used by the frontend service."
  value       = google_service_account.frontend_runtime.email
}

output "public_access_summary" {
  description = "Summary of public access flags used by this blueprint."
  value = {
    frontend_public = var.allow_public_frontend_access
    api_public      = var.allow_public_api_access
  }
}
