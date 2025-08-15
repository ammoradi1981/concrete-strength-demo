# Use Python 3.12
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Copy files
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements_api.txt

# Expose port
EXPOSE 8080

# Run Flask app
CMD ["python", "app.py"]

