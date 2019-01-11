GLOVE_DIR_PATH=$PWD/glove
mkdir $GLOVE_DIR_PATH

wget http://nlp.stanford.edu/data/glove.6B.zip -O $GLOVE_DIR_PATH/glove.6B.zip
unzip $GLOVE_DIR_PATH/glove.6B.zip -d $GLOVE_DIR_PATH

