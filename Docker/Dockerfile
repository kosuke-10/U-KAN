FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

# 非対話モード＆タイムゾーン設定
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Tokyo

RUN apt-get update && \
    apt-get install -y tzdata \
    wget \
    git \
    curl \
    unzip \
    nano \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx && \
    ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone && \
    rm -rf /var/lib/apt/lists/*

# Minicondaインストール
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    bash ~/miniconda.sh -b -p /opt/conda && rm ~/miniconda.sh
ENV PATH="/opt/conda/bin:$PATH"

# GPUライブラリのパス追加
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-11.8/lib64
RUN ln -sf /usr/local/cuda-11.8/lib64/libcusolver.so.11 /usr/local/cuda-11.8/lib64/libcusolver.so.10

# condaの初期設定
RUN conda init bash

# U-KAN用のconda環境作成
RUN conda create -n ukan python=3.10 -y

# 基本パッケージのアップグレード
RUN /bin/bash -c "source /opt/conda/etc/profile.d/conda.sh && conda activate ukan && \
    pip install --upgrade pip setuptools wheel"

# NumPy を最初にインストール（互換性確保）
RUN /bin/bash -c "source /opt/conda/etc/profile.d/conda.sh && conda activate ukan && \
    pip install 'numpy<2.0'"

# PyTorchインストール（CUDA 11.6互換）
RUN /bin/bash -c "source /opt/conda/etc/profile.d/conda.sh && conda activate ukan && \
    pip install torch==1.13.0+cu116 torchvision==0.14.0+cu116 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu116"

# 基本的な科学計算ライブラリ
RUN /bin/bash -c "source /opt/conda/etc/profile.d/conda.sh && conda activate ukan && \
    pip install pandas scipy scikit-learn"

# 画像処理ライブラリ（互換性のあるバージョン）
RUN /bin/bash -c "source /opt/conda/etc/profile.d/conda.sh && conda activate ukan && \
    pip install 'scikit-image>=0.19.0' opencv-python 'pillow>=8.4.0'"

# データ拡張ライブラリ
RUN /bin/bash -c "source /opt/conda/etc/profile.d/conda.sh && conda activate ukan && \
    pip install albumentations==1.1.0"

# 機械学習ライブラリ（互換性のあるバージョン）
RUN /bin/bash -c "source /opt/conda/etc/profile.d/conda.sh && conda activate ukan && \
    pip install 'timm>=0.6.0'"

# その他の必要なライブラリ
RUN /bin/bash -c "source /opt/conda/etc/profile.d/conda.sh && conda activate ukan && \
    pip install addict==2.4.0 pyyaml tqdm tensorboardX typing-extensions yapf==0.31.0"

# 医用画像処理関連
RUN /bin/bash -c "source /opt/conda/etc/profile.d/conda.sh && conda activate ukan && \
    pip install medpy matplotlib seaborn"

# ファイル処理関連
RUN /bin/bash -c "source /opt/conda/etc/profile.d/conda.sh && conda activate ukan && \
    pip install 'tifffile>=2020.9.3'"

# perceptualライブラリ（最後にインストール）
RUN /bin/bash -c "source /opt/conda/etc/profile.d/conda.sh && conda activate ukan && \
    pip install perceptual==0.1"

# デフォルトでukan環境をアクティベート
RUN echo "source /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate ukan" >> ~/.bashrc

# キャッシュクリーンアップ
RUN conda clean -afy && \
    pip cache purge

# コンテナ起動時のコマンド
CMD ["/bin/bash", "-c", "source /opt/conda/etc/profile.d/conda.sh && conda activate ukan && exec bash"]