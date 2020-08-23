import string
import pandas as pd
from nltk import word_tokenize
from nltk.corpus import stopwords
from google_play_scraper import Sort, reviews as reviews_par
from app_store_scraper import AppStore
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
import jieba
import jieba.analyse

# prelims
stop = stopwords.words('english') + list(string.punctuation)
transformer = TfidfTransformer()
cv = CountVectorizer()
jieba.enable_paddle()

# piccollage: 'com.cardinalblue.piccollage.google'
def get_play_store(app_id, n_reviews=200, filter="all"):
    """get play store reviews for app"""
    reviews, _ = reviews_par(
        app_id, count=n_reviews, lang='zh', country='tw',
        sort=Sort.MOST_RELEVANT)
    reviews_df = pd.DataFrame(reviews)
    if filter == "all":
        return reviews_df
    else:
        return reviews_df.query('score' + filter)

# piccollage: 'piccollage-photo-grid-editor'
def get_app_store(app_id, n_reviews=200, filter="all"):
    """get play store reviews for app"""
    data = AppStore(app_name=app_id, country='tw')
    data.review(how_many=n_reviews)
    reviews = data.reviews
    reviews_df = pd.DataFrame(reviews)
    if filter == "all":
        return reviews_df
    else:
        return reviews_df.query('rating' + filter)

def get_features(contents, n_features=50, lang='en'):
    """get top features in accordance to TFIDF"""
    content_ls = []
    for content in contents:
        try:
            if lang == 'zh':
                text_ls = [*jieba.cut(content, use_paddle=True)]
            else:
                text_ls = word_tokenize(content)
                text_ls = [txt.lower() for txt in text_ls if txt.lower() not in stop]
            content_ls.append(' '.join(text_ls))
        except AttributeError:
            pass

    content_vector = cv.fit_transform(content_ls)
    values = transformer.fit_transform(content_vector).todense()
    features = cv.get_feature_names()
    tfidf_df = pd.DataFrame(values, columns=features)
    features_all = tfidf_df.agg('sum')
    min_value = features_all.min()
    max_value = features_all.max()
    scale = lambda x: (x-min_value) / (max_value-min_value) * (100-10) + 10
    features_all = features_all.apply(scale)

    return tfidf_df, features_all.nlargest(n_features)

# get features of negative (ratings <= 3) reviews
play_store = get_play_store('com.cardinalblue.piccollage.google', 10000, '<=3')
app_store = get_app_store('piccollage-photo-grid-editor', 10000, '<=3')

all_reviews = pd.concat([play_store['content'], app_store['review']]).tolist()
all_reviews = [x for x in all_reviews if x]  # remove empty strings

features_matrix, top_features = get_features(all_reviews, lang='zh')
