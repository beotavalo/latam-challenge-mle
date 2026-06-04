terraform {
  # Remote state in GCS so the local bootstrap apply and the CI/CD pipeline share
  # the same state. Backends cannot use variables, so it is configured at init:
  #
  #   terraform init \
  #     -backend-config="bucket=YOUR_TF_STATE_BUCKET" \
  #     -backend-config="prefix=flight-delay-api"
  #
  # No secrets live here; the bucket name is supplied via -backend-config / CI.
  backend "gcs" {}
}
