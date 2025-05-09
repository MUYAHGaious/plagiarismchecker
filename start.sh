#!/bin/bash
set -e

# Run database migrations (if needed)
echo "Running database migrations..."
python -c "from database import Base, engine; Base.metadata.create_all(bind=engine)"

# Download NLTK data (if needed)
echo "Downloading NLTK data..."
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

# Create upload directory if it doesn't exist
mkdir -p uploads

# Start the application
echo "Starting application..."
exec uvicorn main:app --host 0.0.0.0 --port ${PORT:-10000}