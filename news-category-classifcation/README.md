# News Category Classification
This repo contains a simple source code for text-classification based on TextCNN. Corpus is news category dataset in English. Most open sources are a bit difficult to study & make text-classification model for beginners. So, I hope that this repo can be a good solution for people who want to have their own text-classification model.

## Data
News category dataset contains around 200k news headlines from the year 2012 to 2018 obtained from HuffPost. You can download this dataset [here](https://www.kaggle.com/rmisra/news-category-dataset).

The dataset contains 202,372 records. Each json record contains following attributes:

- `category`: Category article belongs to
- `headline`: Headline of the article
- `authors`: Person authored the article
- `link`: Link to the post
- `short_description`: Short description of the article
- `date`: Date the article was published  

Below table shows that the first 5 lines from the dataset provided by [Kaggle](https://www.kaggle.com/).

<p align="left">
<img width="700" src="https://github.com/lyeoni/nlp-tutorial/blob/master/news-category-classifcation/images/data_sample.png">
</p>


## Usage
### 1. Preprocessing corpus
```
structure:
  preprocessing.sh
  ├── tokenization_en.py
      └── remove_emoji.py
  └── fasttext
```
```
$ python tokenization_en.py -h
usage: tokenization_en.py [-h] -input INPUT -column COLUMN -output OUTPUT

optional arguments:
  -h, --help      show this help message and exit
  -input INPUT    data file name to use
  -column COLUMN  column name to use. headline or short_description
  -output OUTPUT  data file name to write
```
example usage:
```
python tokenization_en.py -input News_Category_Dataset_v2.json -column short_description -output news.tk.txt
```
