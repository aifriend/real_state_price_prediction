import re
import string
from pathlib import Path

import emoji
import spacy
from pandas_parallel_apply import DataFrameParallel
from spellchecker import SpellChecker

from src.service.AnalyticService import AnalyticService


class NlpService:
    nlp = spacy.load("en_core_web_sm")
    spell = SpellChecker(distance=2)

    @staticmethod
    def is_english(text):
        if not text:
            return False

        word_list = text.split(' ')
        if not word_list:
            return word_list

        # find those words that may be misspelled
        misspelled = NlpService.spell.unknown(word_list)
        spelled = NlpService.spell.known(word_list)

        ratio = len(spelled) / len(misspelled) if misspelled else 1.0
        if ratio <= 0.7:
            return False

        return True

    @staticmethod
    def lower_case(reviews_df, label):
        reviews_df[label] = reviews_df[label].str.lower()
        return reviews_df

    @staticmethod
    def remove_punctuation(reviews_df, label):
        reviews_df[label] = reviews_df[label].str.replace(string.punctuation, '')
        return reviews_df

    @staticmethod
    def remove_numbers(reviews_df, label):
        reviews_df[label] = reviews_df[label].str.replace('\d+', '')
        return reviews_df

    @staticmethod
    def remove_whitespaces(reviews_df, label):
        reviews_df[label] = reviews_df[label].str.strip()
        return reviews_df

    @staticmethod
    def remove_repeated_punctuation(reviews_df, label):
        reviews_df[label] = reviews_df[label].str.replace(r'(.)\1+', r'\1\1')
        reviews_df[label] = reviews_df[label].str.replace(r'!', r'')
        reviews_df[label] = reviews_df[label].str.replace(r'(\n|\r)', r'')
        return reviews_df

    @staticmethod
    def remove_emoji(reviews_df, label):
        reviews_df[label] = reviews_df[label].apply(
            lambda x: emoji.replace_emoji(x, replace=''))
        return reviews_df

    @staticmethod
    def remove_emoticon_and_symbol(reviews_df, label):
        def removal_func(text):
            emoticon_pattern = re.compile("["
                                          u"\U0001F600-\U0001F64F"  # emoticons
                                          u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                          u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                          u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                          u"\U00002500-\U00002BEF"  # chinese char
                                          u"\U00002702-\U000027B0"
                                          u"\U00002702-\U000027B0"
                                          u"\U000024C2-\U0001F251"
                                          u"\U0001f926-\U0001f937"
                                          u"\U00010000-\U0010ffff"
                                          u"\u2640-\u2642"
                                          u"\u2600-\u2B55"
                                          u"\u200d"
                                          u"\u23cf"
                                          u"\u23e9"
                                          u"\u231a"
                                          u"\ufe0f"  # dingbats
                                          u"\u3030"
                                          "]+", flags=re.UNICODE)
            return emoticon_pattern.sub(r'', text)

        reviews_df[label] = reviews_df[label].apply(removal_func)
        return reviews_df

    @staticmethod
    def remove_special_characters(reviews_df, label):
        # remove_tags_url_email
        # remove_stopwords
        # get word lemma
        # spell check
        reviews_df[label] = DataFrameParallel(
            reviews_df, n_cores=3)[label].apply(NlpService.nlp)
        reviews_df[label] = reviews_df[label].apply(
            lambda x: ' '.join([token.lemma_
                                for token in x
                                if not token.is_stop and
                                not token.like_email and
                                not token.like_url]
                               ))
        return reviews_df

    @staticmethod
    def filter_english_language(reviews_df, label):
        reviews_df = reviews_df[reviews_df[label].apply(NlpService.is_english)]
        return reviews_df

    @staticmethod
    def preprocess_reviews(rev_df):
        # Remove punctuations
        print("Removed punctuations\n")
        rev_df = NlpService.remove_punctuation(rev_df, 'comments')

        # Remove numbers
        print("Removed numbers\n")
        rev_df = NlpService.remove_numbers(rev_df, 'comments')

        # Remove repeated characters
        print("Removed repeated characters\n")
        rev_df = NlpService.remove_repeated_punctuation(rev_df, 'comments')

        # Remove emoticons & symbols
        print("Removed emoticons and symbols\n")
        rev_df = NlpService.remove_emoticon_and_symbol(rev_df, 'comments')

        # Remove tags
        print("Removed special characters\n")
        rev_df = NlpService.remove_special_characters(rev_df, 'comments')

        # Remove whitespaces
        print("Removed whitespaces\n")
        rev_df = NlpService.remove_whitespaces(rev_df, 'comments')

        # Lower case comments
        print("Lowered case\n")
        rev_df = NlpService.lower_case(rev_df, 'comments')

        return rev_df

    @staticmethod
    def process_reviews(reviews_desc_df):
        # reviews usually were submitted with a medium size of 900 characters
        AnalyticService.show_rev_length(reviews_desc_df)
        # select reviews with less than 1000 characters
        reviews_desc_df = reviews_desc_df[
            (250 < reviews_desc_df['comments'].str.len()) &
            (reviews_desc_df['comments'].str.len() <= 260)]
        print(f"Number of reviews lower than 300 characters: {len(reviews_desc_df)}")

        # filter in language
        print("Filter comments in english\n")
        reviews_desc_df = NlpService.filter_english_language(reviews_desc_df, 'comments')

        # Preprocess reviews
        reviews_df = NlpService.preprocess_reviews(reviews_desc_df)

        # Save preprocessed reviews
        reviews_df.to_csv(
            Path.cwd().parents[0].joinpath(
                "data/processed", "neighborhood_reviews.csv.gz"), index=False, compression='gzip')

        return reviews_df
