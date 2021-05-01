# Machine Learning Prediction in Real Time Using Docker and Python REST APIs with Flask

The idea of this article is to do a quick and easy build of a Docker container to perform online inference with trained machine learning models using Python APIs with Flask. Before reading this article, do not hesitate to read Why use Docker for Machine Learning, Quick Install and First Use of Docker, and Build and Run a Docker Container for your Machine Learning Model in which we learn how to use Docker to perform model training and batch inference.
Batch inference is great when you have time to compute your predictions. Let’s imagine you need real time predictions. In this case, batch inference is not more suitable and we need online inference. Many applications would not work or would not be very useful without online predictions such as autonomous vehicles, fraud detection, high-frequency trading, applications based on localization data, object recognition and tracking or brain computer interfaces. Sometimes, the prediction needs to be provided in milliseconds.

To learn this concept, we will implement online inferences (Linear Discriminant Analysis and Multi-layer Perceptron Neural Network models) with Docker and Flask-RESTful.

You can find all the details here: https://xaviervasques.medium.com/machine-learning-prediction-in-real-time-using-docker-and-python-rest-apis-with-flask-4235aa2395eb
