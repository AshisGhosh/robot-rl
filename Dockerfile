FROM nvidia/cuda:12.2.0-devel-ubuntu22.04 

# Set environment variables to non-interactive for installing packages
ENV DEBIAN_FRONTEND=noninteractive

# Install basic dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    python3-dev \
    git \
    wget \
    libgl1 \
    libglib2.0-0 \
    libgl1-mesa-dri \
    libglu1-mesa \
    libxrandr2 \
    libxinerama1 \
    libxi6 \
    libxxf86vm1 \
    xvfb \
    curl \
    python-is-python3 \
    x11-apps \
    && apt-get clean

# Define build argument for app name
ARG APP_NAME=robot-rl
ENV PATH="/root/.local/bin:$PATH"

# Set the working directory
WORKDIR /app

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="/root/.local/bin:$PATH"


# Install Python dependencies using Poetry
RUN poetry config virtualenvs.create false

# Copy only the Poetry files
COPY ${APP_NAME}/pyproject.toml ${APP_NAME}/poetry.lock /app/

RUN poetry install --no-root



# Set the entry point to your training script
CMD ["poetry", "run", "python", "robot_rl/train_robot.py"]
