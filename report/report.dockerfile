FROM ghcr.io/iktovr/diploma-latex-template:latest

# Update sources
RUN apt-get update

# Install extra utilities needed inside the devcontainer.
RUN apt-get install -y \
    clang-format \
    curl \
    git \
    git-lfs \
    htop \
    make \
    tree \
    vim
