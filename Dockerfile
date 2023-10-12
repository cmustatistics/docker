#####################################################
# Dockerfile
#####################################################
# We use a simply base image that only contains python 3.8
# and install Tensorflow 2 without GPU support
# to speed up building time
# In your own projects consider the following.
# For Tensorflow 2 base image with GPU support use:
#   nvcr.io/nvidia/tensorflow:23.07-tf2-py3
# For Pytorch base image with GPU support use:
#   nvcr.io/nvidia/pytorch:23.07-py3
# All Tensorflow images can be found at:
#   https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tensorflow
# Details regarding dependencies installed in each Tensorflow image are available at:
#   https://docs.nvidia.com/deeplearning/deeplearning/frameworks/tensorflow-release-notes/index.html
# All Pytorch images can be found at:
#   https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch
# Details regarding dependencies installed in each Pytorch image are available at:
#   https://docs.nvidia.com/deeplearning/deeplearning/frameworks/pytorch-release-notes/index.html
# Other base images can be found at
#   https://catalog.ngc.nvidia.com/containers
# Note: Nvidia images have a folder at /workspace with documentation about the image
FROM python:3.8-slim-buster

# We install additional packages required for this particular project
# Do indicate which version you need so that the script is determistic. 
# If you do not specify the version, pip will try to install the lastest available
# version that is compatible with your other packages
RUN pip install tensorflow==2.2.0
RUN pip install numpy==1.18.5
RUN pip install tqdm==4.43.0
RUN pip install matplotlib==3.3.0
RUN pip install pandas==1.0.5
RUN apt-get update && apt-get -y install gcc # needed to install GPy
RUN pip install GPy==1.9.9
RUN pip install packaging==20.3
RUN pip install gpflow==2.1.0
RUN pip install protobuf==3.20.*

# We copy the folder containing the source code of our algorithm 
# into the folder /program/src in the container
COPY src /program/src

# If you want to exclude any files inside the above directories
# You should add the exeptions to .dockerignore

# We set the working directory to program
# So that bash starts at /program when, we loggin into the container
WORKDIR /program