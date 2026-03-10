FROM ubuntu:22.04

# Update sources
RUN apt-get update

# Install Utilities
RUN apt-get install -y \
    curl \
    gdb \
    libpq-dev \
    python3 \
    python3-pip \
    clang-format \
    htop \
    tree \
    vim \
    git \
    make \
    wget \
    zip
