FROM python:3.12

# Update sources
RUN apt-get update

# Install Libraries
RUN apt-get install -y \
    cmake \
    ffmpeg \
    gcc \
    g++ \
    libeigen3-dev \
    libopenblas-dev

# Install Python and Tools
RUN apt-get install -y \
    clang-format \
    curl \
    gdb \
    git \
    git-lfs \
    make \
    wget \
    zip

WORKDIR /root/Downloads
# Install AWS
ARG AWS_FILENAME=awscliv2.zip
RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "${AWS_FILENAME}"
RUN unzip "${AWS_FILENAME}" && ./aws/install
RUN rm -rf aws "${AWS_FILENAME}"

# Install Utilities
RUN apt-get install -y \
    htop \
    tree \
    vim
