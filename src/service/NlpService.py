import re
import string

import emoji
import spacy
from pandas import DataFrame
from pandas_parallel_apply import DataFrameParallel
from parallel_pandas import ParallelPandas
from spellchecker import SpellChecker

from src.service.RentalLogger import logger

ParallelPandas.initialize(n_cpu=8, split_factor=4, disable_pr_bar=False)


class NlpService:
    """
    NlpService class

    Methods
    -------
    process_reviews(reviews_df)
    process_listings(listings_df)
    process_listings_with_reviews(listings_df)
    remove_emoticon_and_symbol(reviews_df, label)
    remove_special_characters(reviews_df, label)
    remove_stopwords(reviews_df, label)
    remove_whitespaces(reviews_df, label)
    remove_punctuation(reviews_df, label)
    remove_numbers(reviews_df, label)
    lower_case(reviews_df, label)
    remove_repeated_punctuation(reviews_df, label)
    remove_emoji(reviews_df, label)
    filter_english_language(reviews_df, label)
    """
    nlp = spacy.load("en_core_web_sm")
    spell = SpellChecker(distance=2)

    @staticmethod
    def is_english(text: str) -> bool:
        """
        Check if the text is in english

        Args:
            text : str

        Returns:
            bool
        """
        if not text:
            return False

        word_list = text.split(' ')
        if not word_list:
            return False

        # find those words that may be misspelled
        misspelled = NlpService.spell.unknown(word_list)
        spelled = NlpService.spell.known(word_list)

        ratio = len(spelled) / len(misspelled) if misspelled else 1.0
        if ratio <= 0.8:
            return False

        return True

    @staticmethod
    def lower_case(reviews_df: DataFrame, label: str) -> DataFrame:
        """
        Lower case the text

        Args:
            reviews_df : DataFrame
            label : str

        Returns:
            DataFrame
        """
        reviews_df.loc[:, label] = reviews_df[label].str.lower()
        return reviews_df

    @staticmethod
    def remove_punctuation(reviews_df: DataFrame, label: str) -> DataFrame:
        """
        Remove punctuation from the text

        Args:
            reviews_df : DataFrame
            label : str

        Returns:
            DataFrame
        """
        reviews_df.loc[:, label] = reviews_df[label].str.replace(string.punctuation, '')
        return reviews_df

    @staticmethod
    def remove_numbers(reviews_df, label) -> DataFrame:
        """
        Remove numbers from the text

        Args:
            reviews_df : DataFrame
            label : str

        Returns:
            DataFrame
        """
        reviews_df.loc[:, label] = reviews_df[label].str.replace(r'\d+', '', regex=True)
        return reviews_df

    @staticmethod
    def remove_whitespaces(reviews_df: DataFrame, label: str) -> DataFrame:
        """
        Remove whitespaces from the text

        Args:
            reviews_df : DataFrame
            label : str

        Returns:
            DataFrame
        """
        reviews_df.loc[:, label] = reviews_df[label].str.strip()
        return reviews_df

    @staticmethod
    def remove_repeated_punctuation(reviews_df: DataFrame, label: str) -> DataFrame:
        """
        Remove repeated punctuation from the text

        Args:
            reviews_df : DataFrame
            label : str

        Returns:
            DataFrame
        """
        reviews_df.loc[:, label] = reviews_df[label].str.replace(r'(.)\1+', r'\1\1', regex=True)
        reviews_df.loc[:, label] = reviews_df[label].str.replace(r'!', r'', regex=True)
        reviews_df.loc[:, label] = reviews_df[label].str.replace(r'(\n|\r)', r'', regex=True)
        reviews_df.loc[:, label] = reviews_df[label].str.replace(r'(,)+', r'\1', regex=True)
        return reviews_df

    @staticmethod
    def remove_emoji(reviews_df: DataFrame, label: str) -> DataFrame:
        """
        Remove emoji from the text

        Args:
            reviews_df : DataFrame
            label : str

        Returns:
            DataFrame
        """
        reviews_df.loc[:, label] = reviews_df[label].apply(
            lambda x: emoji.replace_emoji(x, replace=''))
        return reviews_df

    @staticmethod
    def remove_emoticon_and_symbol(reviews_df: DataFrame, label: str) -> DataFrame:
        """
        Remove emoticon and symbol from the text

        Args:
            reviews_df : DataFrame
            label : str

        Returns:
            DataFrame
        """

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

        reviews_df.loc[:, label] = reviews_df[label].apply(removal_func)
        return reviews_df

    @staticmethod
    def remove_special_characters(reviews_df: DataFrame, label: str) -> DataFrame:
        """
        Remove special characters from the text

        Args:
            reviews_df : DataFrame
            label : str

        Returns:
            DataFrame
        """
        # remove_tags_url_email
        # remove_stopwords
        # spell check
        reviews_df.loc[:, label] = DataFrameParallel(
            reviews_df, n_cores=2)[label].apply(NlpService.nlp)
        reviews_df.loc[:, label] = reviews_df[label].apply(
            lambda x: ' '.join([token.text
                                for token in x
                                if not token.is_stop and
                                not token.like_email and
                                not token.like_url]
                               ))
        return reviews_df

    @staticmethod
    def filter_english_language(reviews_df: DataFrame, label: str) -> DataFrame:
        """
        Filter reviews in english

        Args:
            reviews_df : DataFrame
            label : str

        Returns:
            DataFrame
        """
        reviews_df = reviews_df[reviews_df[label].apply(NlpService.is_english)]
        return reviews_df

    @staticmethod
    def process_reviews(reviews_df: DataFrame) -> DataFrame:
        """
        Preprocess reviews

        Args:
            reviews_df : DataFrame

        Returns:
            DataFrame
        """
        # filter in language
        logger.info("Filter comments in english")
        reviews_df = NlpService.filter_english_language(reviews_df, 'comments')

        # Preprocess reviews
        # Remove punctuations
        logger.info("Remove punctuations")
        reviews_df = NlpService.remove_punctuation(reviews_df, 'comments')

        # Remove numbers
        logger.info("Remove numbers")
        reviews_df = NlpService.remove_numbers(reviews_df, 'comments')

        # Remove repeated characters
        logger.info("Remove repeated characters")
        reviews_df = NlpService.remove_repeated_punctuation(reviews_df, 'comments')

        # Remove emoticons & symbols
        logger.info("Remove emoticons and symbols")
        reviews_df = NlpService.remove_emoticon_and_symbol(reviews_df, 'comments')

        # Remove tags
        logger.info("Token filter")
        reviews_df = NlpService.remove_special_characters(reviews_df, 'comments')

        # Remove whitespaces
        logger.info("Remove whitespaces")
        reviews_df = NlpService.remove_whitespaces(reviews_df, 'comments')

        # Remove repeated characters
        logger.info("Remove repeated characters")
        reviews_df = NlpService.remove_repeated_punctuation(reviews_df, 'comments')

        # Lower case comments
        logger.info("Lowered case")
        reviews_df = NlpService.lower_case(reviews_df, 'comments')

        return reviews_df
