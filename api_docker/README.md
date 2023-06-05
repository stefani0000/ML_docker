# A brief description for the project

# in github there're two python scripts, one for ML algorithm and one for the app
- the ML algorithm is a single-layer neural network that uses the sigmoid activation function and loss as binary entropy, it's a model suitable for classification tasks
- the app uses the Flask framework to create a web server that serves predictions using the model; it sets up a web server that accepts a POST request with input data, processes the input using a pre-trained model, and returns the prediction as a JSON response.

# Docker Image

- you can find the docker image here:
[Docker Hub Repository](https://hub.docker.com/r/stefani647464/ml_docker)

# Usage

To use this Docker image, follow the steps below:

1. Install Docker on your machine. Refer to the official Docker documentation for instructions specific to your operating system.

2. Pull the Docker image from Docker Hub using the following command:

docker pull stefani647464/ml_docker

3. Run the Docker container using the pulled image with the desired configurations. 

4. Access the running application in your web browser 

5. When finished, stop the running container using the `docker stop` command. 



