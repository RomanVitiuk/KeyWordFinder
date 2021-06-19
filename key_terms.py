from string import punctuation

from lxml import etree
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer


class KeyWordFinder:
    """This class transform text into array matrix,
    make tokenize, lemmatize, extract stop words, punctuation
    and remove non-noun from that array. Then make vector score
    of each word in array and return five most scored words
    of this text."""
    _STOP = stopwords.words('english')
    _PUNCTUATION = set(punctuation)

    def __init__(self, text):
        self._text = text.lower()

    def _token(self):
        # tokenize and make array from text
        # for example -> [['half', 'a', 'million', 'years', 'ago'],
        #                 ['europe', 'and', 'asia', ',', 'where'],
        #                 ['skull', '(', '3d', 'reconstruction'],
        #                 ]
        return [word_tokenize(x) for x in self._text.split('\n')]

    def _lemm_list(self):
        # lemmatize each word in array
        return [[WordNetLemmatizer().lemmatize(y) for y in x] for
                x in self._token()]

    def _clean_text(self):
        # remove punctuation and stop words from array
        return [[y for y in x if y not in self._STOP and
                 y not in self._PUNCTUATION] for x in self._lemm_list()]

    def _tags_text(self):
        # make tags to each words of array
        # for example -> [[('half', 'NN')],
        #                 [('million', 'CD')],
        #                 [('ago', 'RB')]
        #                 ]
        return [[pos_tag([y]) for y in x] for x in self._clean_text()]

    def _noun(self):
        # remove all non-noun words from array
        # for example -> [[('half', 'NN')],
        #                 [('asia', 'NN')],
        #                 [('aroeira', 'NN')]
        #                 ]
        # and make array matrix from noun word
        # for example -> ['spread europe asia',
        #                 'ha subject intense',
        #                 'skull offering clue'
        #                 ]
        return [' '.join([y[:][0][0] for y in x if y[:][0][1] == 'NN']) for
                x in self._tags_text()]

    def _dict_vectors(self):
        # vectorized each word from sentence in array
        # make dictionary {word: score}
        # for example -> 'fossil': 0.6807108196766997
        # sorted all dictionary by max values
        # return list of five most scored words
        n = len(self._noun())
        vect = TfidfVectorizer()
        mtx = vect.fit_transform(self._noun())
        matrix_score = dict(zip(vect.get_feature_names(), mtx.toarray()[0]))

        for i in range(1, n):
            for k, v in dict(zip(vect.get_feature_names(),
                                 mtx.toarray()[i])).items():
                matrix_score[k] += v

        sorted_list = sorted(matrix_score.items(),
                             key=lambda y: y[1],
                             reverse=True)
        return [x[0] for x in sorted_list[:5]]

    def result(self):
        return self._dict_vectors()


page_content = etree.parse("news.xml").getroot()

for news in page_content.iter():
    if news.get('name') == 'head':
        print(news.text + ':')
    if news.get('name') == 'text':
        key_word = KeyWordFinder(news.text)
        print(*key_word.result())
