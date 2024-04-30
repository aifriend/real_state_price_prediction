from pandas_parallel_apply import DataFrameParallel
from sentence_transformers import SentenceTransformer


class EmbeddingService:

    def __init__(self):
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    def get_embeddings(self, df, text_label):
        df[text_label] = DataFrameParallel(
            df, n_cores=3)[text_label].apply(self.model.encode)
        return df
