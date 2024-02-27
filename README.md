# Docker FastAPI Project

This repository contains a Dockerized FastAPI application for predicting fraudulent activity based on input descriptions.
As this is a job interview there has been no dicussions with the business to know the use case. 
This means that the model might no be the best one for use case, as I have no idea what the use case will be.
If this project is taken over by anyone, please adapt the model accordingly.

## Overview

The project includes the following components:

- **main.py**: Contains the FastAPI application code.
- **Dockerfile**: Defines the Docker image for the FastAPI application.
- **requirements.txt**: Specifies the Python dependencies required for the application.
- **docker-compose.yaml**: Defines the Docker Compose configuration for running the FastAPI application and a web server.
- **trained_rf_model.pkl**: Is the Random Forest model used by the application to predict the fraudulent activity.
- **tfidf_vectorizer.pkl**: Is the encoder model used to encode the description for prediction.

## Prerequisites

To run the FastAPI application locally, you need to have Docker installed (and running) on your system.

## Usage

1. Clone this repository to your local machine:

   ```bash
   git clone https://github.com/your-username/docker_fastapi.git
   ```

2. Navigate to the project directory:

   ```bash
   cd docker_fastapi
   ```

3. Build the Docker image:

   ```bash
   docker-compose up --build
   ```

4. Once the container is running, you can access the FastAPI application at [http://localhost:8000](http://localhost:8000).

## API Documentation

The FastAPI application provides the following endpoints:

- **POST /**: Submit a description and receive a prediction for fraudulent activity.
- **GET /clear**: Clears the history of the descriptions/predictions in memory. You can access it with http://127.0.0.1:8000/clear

## Contributing

Contributions are welcome! If you find any issues or have suggestions for improvement, please open an issue or submit a pull request.


