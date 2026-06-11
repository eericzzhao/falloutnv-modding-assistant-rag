# official Python runtime as parent image
FROM python:3.11-slim

# hugging face spaces requirement
# non-root user to prevent crashes when downloading ML models
RUN useradd -m -u 1000 user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# container's working directory
WORKDIR $HOME/app

# copy the requirements file 
COPY --chown=user backend/requirements.txt .

# install dependencies
# (we use --no-cache-dir to keep the image size small)
RUN pip install --no-cache-dir -r requirements.txt

# copy the backend application code
COPY --chown=user backend/ ./backend/
COPY --chown=user analyze_telemetry.py .
# copy local chunks.pkl file
COPY --chown=user chunks.pkl .

# expose port 7860 for hugging face spaces
EXPOSE 7860

# switch user to non-root before executing
USER user

# command for render port binding
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "7860"]