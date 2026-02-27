#!/bin/bash
set -e
echo "🚀 Starting ChallangeAgentsAi..."

# Wait for dependencies (docker-compose already handles healthcheck, but adding extra safety)
echo "⏳ Waiting for services..."
sleep 5

cd /app

# Initialize database if needed
echo "🗄️  Checking database..."
python scripts/init_db.py 2>/dev/null || echo "⚠️  DB init skipped (may already exist)"

# Wait for Chroma to be ready
echo "⏳ Waiting for Chroma..."
for i in $(seq 1 15); do
    if python -c "import httpx; r = httpx.get('http://${CHROMA_HOST:-chroma}:${CHROMA_PORT:-8000}/api/v2/heartbeat', timeout=2); r.raise_for_status()" 2>/dev/null; then
        echo "✅ Chroma ready"
        break
    fi
    echo "   Chroma not ready yet (attempt $i/15)..."
    sleep 2
done

# Ingest documents if data directory has files
if [ -d "/app/data" ] && [ "$(ls -A /app/data 2>/dev/null)" ]; then
    echo "📚 Ingesting documents to vector DB..."
    python -m vector_db.ingest || echo "⚠️  Ingest failed (check logs above)"
fi

# Start FastAPI in background (port 8000)
echo "🌐 Starting FastAPI backend..."
uvicorn app.main:app --host 0.0.0.0 --port 8000 &

# Start Streamlit (port 8501)
echo "🎨 Starting Streamlit UI..."
exec streamlit run ui/streamlit_app.py --server.port=8501 --server.address=0.0.0.0
