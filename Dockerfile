# official Python runtime as parent image
FROM python:3.11-slim

# container's working directory
WORKDIR /app

# copy the requirements file 
COPY backend/requirements.txt .

# install dependencies
# (we use --no-cache-dir to keep the image size small)
RUN pip install --no-cache-dir -r requirements.txt

# copy the backend application code
COPY backend/ ./backend/
COPY analyze_telemetry.py .

# copy the chroma database directory
COPY vnv_chroma_db/ ./vnv_chroma_db/

# expose port 10000 for the FastAPI server fallback
EXPOSE 10000

# command for render port binding
CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-10000}"]