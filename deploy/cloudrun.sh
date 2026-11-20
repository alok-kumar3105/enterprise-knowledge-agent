#!/usr/bin/env bash
# Deploy to Google Cloud Run with Vertex AI.
#
# Prerequisites: gcloud CLI installed and logged in (`gcloud auth login`),
# a GCP project with billing enabled.
#
#   PROJECT_ID=my-project API_KEY=$(openssl rand -hex 24) ./deploy/cloudrun.sh
set -euo pipefail

: "${PROJECT_ID:?Set PROJECT_ID}"
: "${API_KEY:?Set API_KEY (clients send it as X-API-Key)}"
REGION="${REGION:-us-central1}"
SERVICE="${SERVICE:-knowledge-agent}"
SA_NAME="${SERVICE}-sa"
SA_EMAIL="${SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"

gcloud config set project "$PROJECT_ID"
gcloud services enable run.googleapis.com aiplatform.googleapis.com \
  cloudbuild.googleapis.com secretmanager.googleapis.com artifactregistry.googleapis.com

# Least-privilege service account: it can only call Vertex AI and read its secret.
gcloud iam service-accounts describe "$SA_EMAIL" >/dev/null 2>&1 || \
  gcloud iam service-accounts create "$SA_NAME" --display-name "Knowledge agent runtime"
gcloud projects add-iam-policy-binding "$PROJECT_ID" \
  --member "serviceAccount:${SA_EMAIL}" --role roles/aiplatform.user --condition=None >/dev/null

# Store the API key in Secret Manager instead of a plain env var.
if ! gcloud secrets describe knowledge-agent-api-key >/dev/null 2>&1; then
  printf '%s' "$API_KEY" | gcloud secrets create knowledge-agent-api-key --data-file=-
else
  printf '%s' "$API_KEY" | gcloud secrets versions add knowledge-agent-api-key --data-file=-
fi
gcloud secrets add-iam-policy-binding knowledge-agent-api-key \
  --member "serviceAccount:${SA_EMAIL}" --role roles/secretmanager.secretAccessor >/dev/null

gcloud run deploy "$SERVICE" \
  --source . \
  --region "$REGION" \
  --service-account "$SA_EMAIL" \
  --allow-unauthenticated \
  --memory 1Gi --cpu 1 --concurrency 20 --max-instances 3 \
  --set-env-vars "LLM_PROVIDER=gemini,USE_VERTEX_AI=true,GCP_PROJECT=${PROJECT_ID},GCP_LOCATION=${REGION}" \
  --set-secrets "API_KEY=knowledge-agent-api-key:latest"

URL=$(gcloud run services describe "$SERVICE" --region "$REGION" --format 'value(status.url)')
echo
echo "Deployed: $URL"
echo "Try: curl -s -X POST $URL/ask -H 'Content-Type: application/json' -H \"X-API-Key: \$API_KEY\" \\"
echo "       -d '{\"question\": \"What is the Orion-D warranty and how many Orion-D tickets are open?\"}'"
