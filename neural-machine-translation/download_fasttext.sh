FASTTEXT_DIR_PATH=$PWD/fasttext1

mkdir $FASTTEXT_DIR_PATH

wget https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.vec.gz -O $FASTTEXT_DIR_PATH/cc.en.300.vec.gz
gzip -d $FASTTEXT_DIR_PATH/cc.en.300.vec.gz

wget https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.fr.300.vec.gz -O $FASTTEXT_DIR_PATH/cc.fr.300.vec.gz
gzip -d $FASTTEXT_DIR_PATH/cc.fr.300.vec.gz
