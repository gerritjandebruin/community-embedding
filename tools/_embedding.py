from node2vec import Node2Vec
import pandas as pd
import seaborn as sns
from sklearn.manifold import TSNE

from ._classifier import split


def get_embedding(
    graph, dimensions=128, walk_length=80, num_walks=10, p=1, q=1, weight_key='weight', 
    workers=1, sampling_strategy=None, quiet=False, seed=1, **kwargs
    ):  
    """
    Get the node embedding using Node2Vec. 
    Returns a pd.DataFrame containing the node embedding.
    In the keyword-arguments a pd.Series can be added which will be added to
    the embedding. When target is added in this way, two objects are returned:
    X, y.
    """  
    node2vec = Node2Vec(
        graph, dimensions, walk_length, num_walks, p, q, weight_key, workers, 
        sampling_strategy, quiet, seed=seed
        )
    model = node2vec.fit()
    
    embedding = pd.DataFrame(model.wv.vectors, index=model.wv.index2entity)
    for column, value in kwargs.items():
        embedding[column] = value
    
    return embedding.dropna() # Necessary because not all targets are provided.


def tsne(embedding: pd.DataFrame, n_jobs=-1) -> pd.DataFrame:
    X, y = split(embedding)
    tsne = TSNE(n_jobs=n_jobs).fit_transform(X)
    tsne = pd.DataFrame(tsne, index=X.index, columns=['1st dim', '2nd dim'])
    tsne['target'] = y
    return tsne


def plot_tsne(embedding: pd.DataFrame, n_jobs=-1) -> None:
    sns.scatterplot('1st dim', '2nd dim', 'target', data=tsne(embedding))
    