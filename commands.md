### Build and Run Docker Container for Streamlit Project

To build and run the Docker container for your Streamlit project, follow these steps:

1. **Build the Docker image**:

   ```bash
   docker build -t plant-app .\
   ```

2. **Run the Docker container**:

   ```bash
   docker run -d --name plant-container -p 8501:8501 plant-app
   ```

