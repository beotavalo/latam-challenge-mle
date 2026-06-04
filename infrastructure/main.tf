provider "google" {
  project = var.project_id
  region  = var.region
}

locals {
  required_services = [
    "run.googleapis.com",
    "artifactregistry.googleapis.com",
    "iam.googleapis.com",
    "iamcredentials.googleapis.com",
    "sts.googleapis.com",
    "cloudresourcemanager.googleapis.com",
  ]
}

# Enable the APIs the stack needs.
resource "google_project_service" "enabled" {
  for_each           = toset(local.required_services)
  service            = each.value
  disable_on_destroy = false
}

# Container image registry.
resource "google_artifact_registry_repository" "docker" {
  location      = var.region
  repository_id = var.artifact_repo
  format        = "DOCKER"
  description   = "Container images for the SCL flight-delay API"
  depends_on    = [google_project_service.enabled]
}

# Least-privilege runtime identity for the Cloud Run service.
resource "google_service_account" "runtime" {
  account_id   = "${var.service_name}-run"
  display_name = "Runtime SA for ${var.service_name}"
}

# Identity assumed by GitHub Actions (via WIF) to build and deploy.
resource "google_service_account" "deployer" {
  account_id   = "${var.service_name}-deployer"
  display_name = "GitHub Actions deployer for ${var.service_name}"
}

resource "google_project_iam_member" "deployer_roles" {
  for_each = toset([
    "roles/run.admin",               # deploy/update the Cloud Run service
    "roles/artifactregistry.writer", # push images
    "roles/iam.serviceAccountUser",  # act as the runtime SA
  ])
  project = var.project_id
  role    = each.value
  member  = "serviceAccount:${google_service_account.deployer.email}"
}

# ---------------------------------------------------------------------------
# Workload Identity Federation — keyless auth for GitHub Actions (no SA key
# stored as a secret). The OIDC token from GitHub is exchanged for short-lived
# credentials that impersonate the deployer SA, restricted to this repository.
# ---------------------------------------------------------------------------
resource "google_iam_workload_identity_pool" "github" {
  workload_identity_pool_id = "${var.service_name}-gh-pool"
  display_name              = "GitHub Actions pool"
  depends_on                = [google_project_service.enabled]
}

resource "google_iam_workload_identity_pool_provider" "github" {
  workload_identity_pool_id          = google_iam_workload_identity_pool.github.workload_identity_pool_id
  workload_identity_pool_provider_id = "github-oidc"
  display_name                       = "GitHub OIDC"

  attribute_mapping = {
    "google.subject"       = "assertion.sub"
    "attribute.repository" = "assertion.repository"
  }
  # Only tokens from our repository may use this provider.
  attribute_condition = "assertion.repository == \"${var.github_repository}\""

  oidc {
    issuer_uri = "https://token.actions.githubusercontent.com"
  }
}

# Let workflows from our repo impersonate the deployer SA.
resource "google_service_account_iam_member" "wif_deployer" {
  service_account_id = google_service_account.deployer.name
  role               = "roles/iam.workloadIdentityUser"
  member             = "principalSet://iam.googleapis.com/${google_iam_workload_identity_pool.github.name}/attribute.repository/${var.github_repository}"
}

# ---------------------------------------------------------------------------
# Cloud Run service.
# ---------------------------------------------------------------------------
resource "google_cloud_run_v2_service" "api" {
  name     = var.service_name
  location = var.region
  ingress  = "INGRESS_TRAFFIC_ALL"

  template {
    service_account = google_service_account.runtime.email

    scaling {
      min_instance_count = var.min_instances
      max_instance_count = var.max_instances
    }

    containers {
      image = var.image

      ports {
        container_port = 8080
      }

      resources {
        limits = {
          cpu    = var.cpu
          memory = var.memory
        }
      }

      env {
        name  = "RATE_LIMIT"
        value = var.rate_limit
      }
    }
  }

  depends_on = [google_project_service.enabled]

  # CI/CD updates the image on every deploy; ignore drift on bootstrap default.
  lifecycle {
    ignore_changes = [client, client_version]
  }
}

# Public endpoint: required so the airport team and the stress test can reach it.
resource "google_cloud_run_v2_service_iam_member" "public" {
  name     = google_cloud_run_v2_service.api.name
  location = var.region
  role     = "roles/run.invoker"
  member   = "allUsers"
}
