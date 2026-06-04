# Infrastructure (Terraform → GCP Cloud Run)

Infrastructure-as-code for deploying the SCL flight-delay API to **Google Cloud
Run**, with **Artifact Registry**, least-privilege service accounts, and
**Workload Identity Federation** so GitHub Actions deploys **without any
long-lived service-account key**.

```
GitHub Actions ──OIDC──▶ Workload Identity Pool ──impersonate──▶ deployer SA
                                                                    │
        build image ▶ Artifact Registry ◀── push ──────────────────┤
                                                                    ▼
                                                  Cloud Run service (public)
                                                   runs as runtime SA
```

## Files
| File | Purpose |
|------|---------|
| `versions.tf` | Terraform + provider version pins |
| `backend.tf`  | GCS remote state (configured at `init` time) |
| `variables.tf`| Inputs (project, region, image, rate limit, scaling…) |
| `main.tf`     | APIs, Artifact Registry, SAs + IAM, WIF, Cloud Run |
| `outputs.tf`  | URL + values needed as GitHub secrets |
| `terraform.tfvars.example` | Copy → `terraform.tfvars` |

## One-time bootstrap (run locally by a project owner)

```bash
# 0) Auth + a GCS bucket for Terraform state (one per project).
gcloud auth application-default login
gcloud storage buckets create gs://<TF_STATE_BUCKET> --location=us-central1

# 1) Init with the remote state backend.
cd infrastructure
terraform init \
  -backend-config="bucket=<TF_STATE_BUCKET>" \
  -backend-config="prefix=flight-delay-api"

# 2) Apply. The image defaults to a public placeholder, so this succeeds before
#    any image exists and creates the registry + WIF + SAs + Cloud Run.
terraform apply \
  -var="project_id=<PROJECT_ID>" \
  -var="github_repository=beotavalo/latam-challenge-mle"

# 3) Read the outputs for the GitHub secrets below.
terraform output
```

## GitHub Actions secrets / variables (provided by the repo owner)

The CD pipeline (LT-MLE-006) reads these — **no values are committed**:

| Name | Source |
|------|--------|
| `GCP_PROJECT_ID` | your project id |
| `GCP_REGION` | e.g. `us-central1` |
| `GCP_WORKLOAD_IDENTITY_PROVIDER` | `terraform output workload_identity_provider` |
| `GCP_DEPLOYER_SA` | `terraform output deployer_service_account` |
| `TF_STATE_BUCKET` | the bucket created above |

WIF is keyless, so none of these is a credential file. (A service-account JSON
key stored as `GCP_SA_KEY` is the fallback if WIF can't be used.)

### One extra grant for CD (state bucket access)
The state bucket is created out-of-band (Step 0), so grant the deployer SA
access to it once — the CD `terraform init/apply` reads and writes state there:

```bash
gcloud storage buckets add-iam-policy-binding gs://<TF_STATE_BUCKET> \
  --member="serviceAccount:$(terraform output -raw deployer_service_account)" \
  --role="roles/storage.objectAdmin"
```

## How CD deploys (LT-MLE-006)
1. Authenticate to GCP via WIF (`google-github-actions/auth`).
2. Build the image and push to Artifact Registry.
3. `terraform apply -var="image=<REGION>-docker.pkg.dev/<PROJECT>/flight-delay/flight-delay-api:<sha>"`
   to roll the new image onto Cloud Run.
4. Publish `terraform output cloud_run_url` (used to update `Makefile` line 26).

## Notes
- `rate_limit` is empty by default so `make stress-test` can saturate the API;
  set it (e.g. `"120/minute"`) for the public week.
- `max_instances` is the real overload/cost guardrail; `min_instances = 0`
  scales to zero when idle.
