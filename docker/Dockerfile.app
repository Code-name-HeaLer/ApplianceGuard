FROM python:3.9-slim

WORKDIR /app

# Install system build dependencies (needed for some python libs)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Install the package
RUN pip install -e .

# Expose Streamlit port
EXPOSE 8501

# Run the app
CMD ["streamlit", "run", "dashboard/app.py", "--server.address=0.0.0.0"]