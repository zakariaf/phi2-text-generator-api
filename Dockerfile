# Use an official Python runtime as a parent image
FROM python:3.8-slim-buster

# Set the working directory in the container
WORKDIR /usr/src/app

# Install PyTorch
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Copy the requirements file into the container at /usr/src/app
COPY requirements.txt ./

# Install any needed packages specified in requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy the rest of your application's code into the container
COPY . .


EXPOSE 9001

ENV FLASK_APP=phi2_demo.py
ENV HF_HOME=/model_cache

CMD ["flask", "run", "--host=0.0.0.0", "--port=9001"]
