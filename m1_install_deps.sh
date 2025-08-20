#!/bin/bash
# 安装macOS系统依赖
brew update
brew install cmake pkg-config wget

# 创建虚拟环境（建议使用Python 3.9+）
python3 -m venv face_aesthetics_env
source face_aesthetics_env/bin/activate

# 安装PyPI依赖
pip install numpy opencv-python matplotlib pandas scipy

# 安装dlib特殊处理
pip install --no-binary :all: dlib

# 下载人脸特征点模型
mkdir -p models
wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2 -P models/
bunzip2 models/shape_predictor_68_face_landmarks.dat.bz2

echo "安装完成！激活环境使用：source face_aatures_env/bin/activate"
