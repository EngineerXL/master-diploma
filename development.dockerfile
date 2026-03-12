FROM python:3.12

# Update sources
RUN apt-get update

# Install Libraries
RUN apt-get install -y \
    cmake \
    ffmpeg \
    gcc \
    g++ \
    libceres-dev \
    libeigen3-dev

# Install Python and Tools
RUN apt-get install -y \
    clang-format \
    curl \
    gdb \
    git \
    make \
    wget \
    zip

# Install AWS
WORKDIR /root/Downloads
RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
RUN unzip awscliv2.zip && ./aws/install
RUN rm -rf aws "awscliv2.zip"

# Install Utilities
RUN apt-get install -y \
    htop \
    tree \
    vim
