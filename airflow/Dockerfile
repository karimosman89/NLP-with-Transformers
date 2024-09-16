# Airflow Dockerfile
FROM apache/airflow:2.3.0

# Copy your DAGs (if any)
COPY dags /opt/airflow/dags/

# Install additional Python packages
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Expose the webserver port
EXPOSE 8080