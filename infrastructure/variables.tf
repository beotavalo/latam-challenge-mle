variable "project_id" {
  type        = string
  description = "GCP project ID to deploy into."
}

variable "region" {
  type        = string
  description = "GCP region for Artifact Registry and Cloud Run."
  default     = "us-central1"
}

variable "service_name" {
  type        = string
  description = "Cloud Run service name (also used to derive SA and pool names)."
  default     = "flight-delay-api"
}

variable "artifact_repo" {
  type        = string
  description = "Artifact Registry Docker repository ID."
  default     = "flight-delay"
}

variable "github_repository" {
  type        = string
  description = "owner/repo allowed to deploy via Workload Identity Federation, e.g. beotavalo/latam-challenge-mle."
}

variable "image" {
  type        = string
  description = "Container image to deploy. Defaults to a public placeholder so the first (bootstrap) apply succeeds before any image is pushed; CI/CD overrides it with the built image."
  default     = "us-docker.pkg.dev/cloudrun/container/hello"
}

variable "rate_limit" {
  type        = string
  description = "Per-IP rate limit passed to the app as RATE_LIMIT (e.g. \"120/minute\"). Empty disables it (recommended while running the stress test)."
  default     = ""
}

variable "min_instances" {
  type        = number
  description = "Minimum Cloud Run instances. 0 = scale to zero (cheapest)."
  default     = 0
}

variable "max_instances" {
  type        = number
  description = "Maximum Cloud Run instances — the main overload/cost guardrail."
  default     = 3
}

variable "cpu" {
  type        = string
  description = "CPU limit per Cloud Run instance."
  default     = "1"
}

variable "memory" {
  type        = string
  description = "Memory limit per Cloud Run instance."
  default     = "512Mi"
}
