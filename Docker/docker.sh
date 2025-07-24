# ----------------
# docker.sh
# dockerimageの作成
# dockerコンテナの作成
# docker.sh "GPU番号"
# ----------------


# -----------------------------
# 設定（必要に応じて書き換えてOK）
# -----------------------------

# GPU番号が指定されていなければエラーで終了
if [ -z "$1" ]; then
  echo "エラー: GPU番号を指定してください"
  echo "使い方: bash docker.sh <GPU番号>"
  exit 1
fi


GPU_ID=$1
IMAGE_NAME=yoshida/u-kan
CONTAINER_NAME=yoshida_u-kan${GPU_ID}

HOST_DIR=/home/yoshida/U-KAN
CONTAINER_DIR=/U-KAN

docker build --force-rm -t ${IMAGE_NAME} .

docker run --gpus "device=${GPU_ID}" \
   --shm-size=24g \
    -it -v ${HOST_DIR}:${CONTAINER_DIR} \
    --name ${CONTAINER_NAME} ${IMAGE_NAME} \

