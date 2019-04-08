# Install Mecab library for korean natural language pre-proocessing on Google Cloud Platform(GCP) operated by Ubuntu 16.03 LTS 


# Remove python 2.x, and related packages.
# The following packages were automatically installed on GCP and are no longer required: libpython-all-dev, libpython-dev, libpython-stdlib, libpython2.7, libpython2.7-dev, libpython2.7-minimal, libpython2.7-stdlib
# refer to http://egloos.zum.com/chanik/v/4112679
sudo apt purge python2.7-minimal
update-alternatives --install /usr/bin/python python /usr/bin/python3.5 1
# Run the code below to check it works well
# $ python --version
# $ sudo python --version
# Python 3.5.3


# Python Development Environment Settings
# refer to https://cloud.google.com/python/setup?hl=ko
sudo apt update
sudo apt install python3 python3-dev
wget https://bootstrap.pypa.io/get-pip.py
sudo python get-pip.py
# Run the code below to check it works well
# $ pip --version
# pip 19.0.3 from /usr/local/lib/python3.5/dist-packages/pip (python 3.5)


# Install KoNLPy, Mecab
# refer to https://medium.com/@juneoh/windows-10-64bit-%E1%84%8B%E1%85%A6%E1%84%89%E1%85%A5-pytorch-konlpy-mecab-%E1%84%89%E1%85%A5%E1%86%AF%E1%84%8E%E1%85%B5%E1%84%92%E1%85%A1%E1%84%80%E1%85%B5-4af8b049a178
sudo apt-get update && sudo apt-get install -y openjdk-8-jdk g++ build-essential autoconf automake
sudo pip install jpype1 konlpy
bash <(curl -s https://raw.githubusercontent.com/konlpy/konlpy/master/scripts/mecab.sh)

# Install mecab-ko-dic.
cd /tmp
curl -LO https://bitbucket.org/eunjeon/mecab-ko-dic/downloads/mecab-ko-dic-2.0.1-20150920.tar.gz
tar -zxf mecab-ko-dic-2.0.1-20150920.tar.gz
cd mecab-ko-dic-2.0.1-20150920
./autogen.sh
./configure
make
sudo sh -c 'echo "dicdir=/usr/local/lib/mecab/dic/mecab-ko-dic" > /usr/local/etc/mecabrc'
sudo make install
sudo ldconfig

# After installation, run the code below to check it works well
# $ python
# >>> from konly.tag import Mecab
# >>> mecab = Mecab()
# >>> mecab.morphs('your sentence')
