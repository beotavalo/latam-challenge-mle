output "cloud_run_url" {
  description = "Public HTTPS URL of the deployed API."
  value       = google_cloud_run_v2_service.api.uri
}

output "artifact_registry_repo" {
  description = "Docker repository path for pushing images."
  value       = "${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.docker.repository_id}"
}

output "workload_identity_provider" {
  description = "Full resource name of the WIF provider (GitHub secret GCP_WORKLOAD_IDENTITY_PROVIDER)."
  value       = google_iam_workload_identity_pool_provider.github.name
}

output "deployer_service_account" {
  description = "Deployer SA email impersonated by GitHub Actions (GitHub secret GCP_DEPLOYER_SA)."
  value       = google_service_account.deployer.email
}
