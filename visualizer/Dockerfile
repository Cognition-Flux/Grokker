# Use the Miniconda3 image as the base
FROM continuumio/miniconda3

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.yml .


# Create conda environment and install dependencies
RUN conda env create -f requirements.yml

# Make RUN commands use the new environment

SHELL ["conda", "run", "-n", "groker", "/bin/bash", "-c"]

COPY app/ ./app/

# Expose port 8080 for the Streamlit app
EXPOSE 8085


# Set environment variables for logging
ENV PYTHONUNBUFFERED=1
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_LOG_LEVEL=info

# Set up the entrypoint
ENTRYPOINT ["conda","run", "--no-capture-output", "-n", "groker", "streamlit", "run", "app/front.py", "--server.port=8085", "--server.address=0.0.0.0", "--logger.level=info"]
