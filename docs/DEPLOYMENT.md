# Deployment Guide

## Azure App Service (Backend API)

### Prerequisites
- Azure CLI installed and authenticated
- Resource group created

### Deploy via CLI

```bash
# Create resource group
az group create --name fitform-rg --location eastus

# Deploy backend
cd fitform-ai
az webapp up \
    --name fitform-ai-api \
    --resource-group fitform-rg \
    --runtime "PYTHON:3.11" \
    --sku B1 \
    --src-path backend/

# Set environment variables
az webapp config appsettings set \
    --name fitform-ai-api \
    --resource-group fitform-rg \
    --settings \
        AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com/" \
        AZURE_OPENAI_API_KEY="your-key" \
        AZURE_OPENAI_DEPLOYMENT="gpt-4o"
```

### Deploy via Docker

```bash
# Build and push to Azure Container Registry
az acr create --name fitformai --resource-group fitform-rg --sku Basic
az acr build --registry fitformai --image fitform-backend:latest -f Dockerfile.backend .

# Deploy to Container Apps
az containerapp create \
    --name fitform-api \
    --resource-group fitform-rg \
    --image fitformai.azurecr.io/fitform-backend:latest \
    --target-port 8000 \
    --ingress external
```

## CI/CD with GitHub Actions

The included `.github/workflows/ci.yml` pipeline runs on every push to `main`:

1. **Lint** — ruff check and format
2. **Test** — pytest for backend and edge modules
3. **Docker** — Build and smoke-test the container image

To add automatic Azure deployment, add these secrets to your GitHub repo:
- `AZURE_CREDENTIALS` (service principal JSON)
- `AZURE_OPENAI_ENDPOINT`
- `AZURE_OPENAI_API_KEY`

## Edge Device Configuration

Update the Jetson's `.env` to point to the deployed backend:

```
BACKEND_URL=https://fitform-ai-api.azurewebsites.net
```
