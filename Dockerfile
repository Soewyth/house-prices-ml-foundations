FROM python:3.12-slim

# Install system dependencies and clean up apt cache to reduce image size
RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*

# Copy the project code into the container
WORKDIR /app
COPY . /app/

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    pip install -e .

# Expose the port for the FastAPI app
EXPOSE 8000

# Command to run the FastAPI app using uvicorn
CMD ["uvicorn", "house_prices_ml_foundations.api.app:app", "--host", "0.0.0.0", "--port", "8000"]