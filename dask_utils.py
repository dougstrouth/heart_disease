import dask.dataframe as dd
from dask.distributed import Client, LocalCluster
import coiled

def get_dask_client(cluster_type='local', **kwargs):
    """
    Initializes and returns a Dask client for either a local or Coiled cluster.

    Args:
        cluster_type (str): Type of Dask cluster ('local' or 'coiled').
        **kwargs: Additional keyword arguments for cluster initialization.
    Returns:
        dask.distributed.Client: A Dask client instance.
    """
    if cluster_type == 'local':
        cluster = LocalCluster(n_workers=4, threads_per_worker=2, memory_limit='1.5GB', **kwargs)
        client = Client(cluster)
    elif cluster_type == 'coiled':
        cluster = coiled.Cluster(n_workers=4, **kwargs)
        client = cluster.get_client()
    else:
        raise ValueError("cluster_type must be 'local' or 'coiled'")

    return client

def dask_dataframe_from_pandas(df, npartitions=None):
    """
    Converts a Pandas DataFrame to a Dask DataFrame.

    Args:
        df (pandas.DataFrame): The Pandas DataFrame to convert.
        npartitions (int, optional): The number of partitions for the Dask DataFrame. 
                                     If None, Dask will choose a suitable number.

    Returns:
        dask.dataframe.DataFrame: The converted Dask DataFrame.
    """
    return dd.from_pandas(df, npartitions=npartitions)

def get_dask_dataframe_statistics(ddf):
    """
    Computes basic descriptive statistics for a Dask DataFrame.

    Args:
        ddf (dask.dataframe.DataFrame): The Dask DataFrame.

    Returns:
        pandas.DataFrame: A Pandas DataFrame containing the descriptive statistics.
    """
    return ddf.describe().compute()