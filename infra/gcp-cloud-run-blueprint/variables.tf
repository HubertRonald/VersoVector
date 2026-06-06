variable "project_id" {
  description = "Generic Google Cloud project ID used for this sanitized blueprint. Do not commit real production values."
  type        = string
}

variable "region" {
  description = "Google Cloud region for Artifact Registry and Cloud Run services."
  type        = string
  default     = "us-central1"
}

variable "resource_prefix" {
  description = "Prefix used to name public blueprint resources."
  type        = string
  default     = "versovector"
}

variable "enable_project_services" {
  description = "Whether Terraform should enable required Google Cloud APIs. Set to false if APIs are managed elsewhere."
  type        = bool
  default     = false
}

variable "artifact_registry_repository_id" {
  description = "Artifact Registry repository ID for Docker images."
  type        = string
  default     = "versovector-services"
}

variable "api_image" {
  description = "Container image URI for the VersoVector FastAPI service. Example: REGION-docker.pkg.dev/PROJECT_ID/REPO/versovector-api:TAG"
  type        = string
}

variable "frontend_image" {
  description = "Container image URI for the VersoVector Gradio frontend. Example: REGION-docker.pkg.dev/PROJECT_ID/REPO/versovector-frontend:TAG"
  type        = string
}

variable "api_container_port" {
  description = "Container port exposed by the API image."
  type        = number
  default     = 8001
}

variable "frontend_container_port" {
  description = "Container port exposed by the frontend image."
  type        = number
  default     = 7860
}

variable "api_cpu" {
  description = "CPU limit for the API Cloud Run service."
  type        = string
  default     = "2"
}

variable "api_memory" {
  description = "Memory limit for the API Cloud Run service. Model loading may require more memory than a normal web service."
  type        = string
  default     = "4Gi"
}

variable "frontend_cpu" {
  description = "CPU limit for the frontend Cloud Run service."
  type        = string
  default     = "1"
}

variable "frontend_memory" {
  description = "Memory limit for the frontend Cloud Run service."
  type        = string
  default     = "1Gi"
}

variable "api_min_instances" {
  description = "Minimum API instances. Keep 0 for a cost-conscious public blueprint."
  type        = number
  default     = 0
}

variable "api_max_instances" {
  description = "Maximum API instances for this public blueprint."
  type        = number
  default     = 2
}

variable "frontend_min_instances" {
  description = "Minimum frontend instances. Keep 0 for a cost-conscious public blueprint."
  type        = number
  default     = 0
}

variable "frontend_max_instances" {
  description = "Maximum frontend instances for this public blueprint."
  type        = number
  default     = 2
}

variable "allow_public_frontend_access" {
  description = "Whether to allow unauthenticated public access to the frontend Cloud Run service."
  type        = bool
  default     = true
}

variable "allow_public_api_access" {
  description = "Whether to allow unauthenticated public access to the API Cloud Run service. Keep false for the target private-service architecture."
  type        = bool
  default     = false
}

variable "api_timeout_seconds" {
  description = "Timeout configured in the frontend for API calls."
  type        = number
  default     = 300
}

variable "model_bundle_uri" {
  description = "Placeholder URI for a future remote model bundle location, such as GCS or an MLflow artifact store. Empty by default for this blueprint."
  type        = string
  default     = ""
}

variable "labels" {
  description = "Common labels applied to blueprint resources."
  type        = map(string)
  default = {
    app         = "versovector"
    environment = "blueprint"
    visibility  = "public-sanitized"
  }
}
