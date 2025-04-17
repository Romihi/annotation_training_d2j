# Use the L4T (Linux for Tegra) base image which is compatible with Jetson
FROM nvcr.io/nvidia/l4t-base:r35.2.1

# Install system dependencies for PyQt
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    python3-pyqt5 \
    python3-pyqt5.qtsvg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libxcb-xinput0 \
    libxkbcommon-x11-0 \
    libfontconfig1 \
    libxcb-xinerama0 \
    libdbus-1-3 \
    libxcb-image0 \
    libxcb-icccm4 \
    libxcb-keysyms1 \
    libxcb-render-util0 \
    libxcb-xkb1 \
    xauth \
    xvfb \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
RUN apt-get update && apt-get install -y fonts-takao

WORKDIR /app

# 上の階層のファイルを全部コピー
COPY . .
# Install Python dependencies (create a requirements.txt with your dependencies)
COPY docker/requirements.txt /app/
RUN apt-get update && apt-get install -y python3-pyqt5
RUN pip3 install --no-cache-dir -r requirements.txt
RUN pip3 install --no-cache-dir --upgrade numpy 

# Copy your application code
COPY . /app/

# Set the display environment variable
ENV DISPLAY=:0

# Command to run your application
CMD ["python3", "annotation_bbox.py"]
