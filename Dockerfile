# official Python runtime as parent image
FROM python:3.11-slim

# container's wokring directory
WORKDIR /app

# copy the requirements file 
COPY backend/requirements.txt .

# install dependencies
# (we use --no-cache-dir to keep the image size small)
RUN pip install --no-cache-dir -r requirements.txt

# copy the rest of the backend application code
COPY backend/main.py .
COPY backend/ ./backend/
COPY analyze_telemetry.py .
# copy the chroma daatbase directory
COPY vnv_chroma_db/ ./vnv_chroma_db/

# expose port 800 for the FastAPI server
EXPOSE 8000

# uvicorn command
#CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
# command for render
#CMD uvicorn backend.main:app --host 0.0.0.0 --port $PORT
CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-10000}"]