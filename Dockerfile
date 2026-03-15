FROM python:3.12-slim

LABEL maintainer="David Girard"
LABEL description="AI File Discovery — Find, classify, and monitor AI/ML files"

WORKDIR /app

# Copy only what's needed (no test data, no caches)
COPY aifiles.py server.py generate_test_data.py pyproject.toml README.md ./
COPY tests/ tests/

# Install the package
RUN pip install --no-cache-dir .

# Create volume mount point
RUN mkdir /data

# Generate sample test data for demo
RUN python3 generate_test_data.py

EXPOSE 8505

# Default: start the web dashboard scanning /data
CMD ["aifiles-server", "--port", "8505", "--scan-path", "/data"]
