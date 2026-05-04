FROM python:3.12-slim

WORKDIR /app

# Copy the requirements and install them
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project files
COPY . .

# Expose the port Streamlit uses
EXPOSE 8501

# Start the dashboard by default
CMD ["streamlit", "run", "dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]