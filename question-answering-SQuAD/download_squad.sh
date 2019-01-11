SQUAD_DIR_PATH=$PWD/squad

mkdir $SQUAD_DIR_PATH

wget https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json -O $SQUAD_DIR_PATH/train-v1.1.json
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json -O $SQUAD_DIR_PATH/dev-v1.1.json



