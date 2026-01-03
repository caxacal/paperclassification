# PSITE Abstract Classifier API

Flask API for classifying academic paper abstracts using SciBERT.

## Deployment

1. Upload model files to `/model` directory
2. Set `API_KEY` environment variable
3. Deploy on Render.com

## Endpoints

- `GET /` - Service info
- `GET /health` - Health check
- `POST /classify` - Classify abstract