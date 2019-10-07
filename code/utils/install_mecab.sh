# install mecab-ko on Ubuntu
# refer to https://medium.com/@juneoh/windows-10-64bit-%E1%84%8B%E1%85%A6%E1%84%89%E1%85%A5-pytorch-konlpy-mecab-%E1%84%89%E1%85%A5%E1%86%AF%E1%84%8E%E1%85%B5%E1%84%92%E1%85%A1%E1%84%80%E1%85%B5-4af8b049a178
bash <(curl -s https://raw.githubusercontent.com/kh-kim/nlp_preprocessing/master/setup.WSL.sh)

# After installation, run the code below to check it works well
# $ python
# >>> from konlpy.tag import Mecab
# >>> mecab = Mecab()
# >>> mecab.morphs('your sentence')

