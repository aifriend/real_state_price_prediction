from pandas import DataFrame
from pandas_parallel_apply import DataFrameParallel
from sentence_transformers import SentenceTransformer


class EmbeddingService:
    """
    The class provides methods to generate embeddings for the text data.
    The embeddings are generated using Sentence Transformers
    (https://github.com/UKPLab/sentence-transformers)
    and stored in a new column 'comments_emb'
    """

    def __init__(self):
        """
        Initialize the model
        """
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    def get_embeddings(self, df: DataFrame, text_label: str) -> DataFrame:
        """
        Generate embeddings for the text data

        Parameters
        ----------
        df : DataFrame
            The dataframe containing the text data
        text_label : str
            The name of the column containing the text data

        Returns
        -------
        DataFrame
            The dataframe with the new column 'comments_emb'
        """
        df['comments_emb'] = DataFrameParallel(
            df, n_cores=2)[text_label].apply(self.model.encode)
        return df
