# Use Python 3.10 slim image
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /code

# Copy the requirements.txt and install dependencies
COPY ./requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the source code into the container
COPY ./src ./src

# Copy the trained model into the container
COPY ./src/trained_model.keras /code/trained_model.keras

# Default command to run the Streamlit app
CMD ["streamlit", "run", "src/main.py"]
