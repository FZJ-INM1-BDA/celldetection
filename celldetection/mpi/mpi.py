"""
Practical Examples
------------------

In the example below a ``server`` distributes ``data`` to ``workers``. The workers process each data item individually
and send results back to the server. After all data has been processed and send back, the server has all results
in correct order.

>>> from celldetection import mpi
... import numpy as np
...
... comm, rank, ranks = mpi.get_comm(return_ranks=True)
... assert ranks >= 2
...
... server = 0
... workers = set(range(ranks)) - {server}
...
... data = range(10)
...
... # Server
... if rank == server:
...     results = mpi.serve(comm, iterator=data, ranks=workers)
...     assert np.allclose(results, np.array(data) ** 2)  # results are in correct order
...     print(results, flush=True)
...
... # Worker
... else:
...     for idx, item in mpi.query(comm, server):
...         result = item ** 2  # process item
...         mpi.send(comm, result, server, tag=idx)  # send result to server


In the example below a ``server`` distributes ``data`` to ``workers``. The workers process each data item individually
and send results to ``sink``. The sink can handle results individually on demenad. In this example results are simply
collected and sorted.


>>> from celldetection import mpi
... import numpy as np
...
... comm, rank, ranks = mpi.get_comm(return_ranks=True)
... assert ranks >= 3
...
... server = 0
... sink = 1
... workers = set(range(ranks)) - {server, sink}
...
... data = range(10)
...
... # Server
... if rank == server:
...     mpi.serve(comm, iterator=data, ranks=workers)
...
... # Sink
... elif rank == sink:
...     indices, results = [], []
...     for idx, item in mpi.sink(comm, ranks=workers):
...         indices.append(indices), results.append(item)  # handle result
...     results = [x for _, x in sorted(zip(indices, results))]  # sort by index
...     assert np.allclose(results, np.array(data) ** 2)  # results are in correct order
...     print(results, flush=True)
...
... # Worker
... else:
...     for idx, item in mpi.query(comm, server, forward_stop_signal=sink):
...         result = item ** 2  # process item
...         mpi.send(comm, result, sink, tag=idx)  # send result to sink


MPI Operations
--------------
"""


import numpy as np

_ERR = None
try:
    from mpi4py import MPI

    ANY_TAG = MPI.ANY_TAG
    ANY_SOURCE = MPI.ANY_SOURCE
except ModuleNotFoundError as e:
    _ERR = e
    MPI = False
    ANY_TAG = -1
    ANY_SOURCE = -2  # may differ depending on MPI implementation

__all__ = ['recv', 'send', 'sink', 'query', 'serve', 'all_filter', 'get_local_comm', 'get_hosts', 'get_comm']


def assert_mpi(func):
    def func_wrapper(*a, **k):
        if not MPI:
            raise ModuleNotFoundError(
                f'In order to use mpi functions, MPI must be installed. Could not import mpi4py.\n\n'
                f'Check out: https://mpi4py.readthedocs.io/en/stable/install.html\n\n{str(_ERR)}')
        return func(*a, **k)

    func_wrapper.__doc__ = func.__doc__
    return func_wrapper


@assert_mpi
def get_hosts(comm, return_ranks=False):
    host = MPI.Get_processor_name()
    hosts = list(np.sort(np.unique(comm.allgather(host))))
    res = host, hosts
    if return_ranks:
        node_rank = next(i for i, h in enumerate(hosts) if h == host)
        node_ranks = len(list(hosts))
        res += (node_rank, node_ranks)
    return res


@assert_mpi
def get_comm(comm=None, return_ranks=True):
    comm = comm or MPI.COMM_WORLD
    res = comm,
    if return_ranks:
        rank = comm.Get_rank()
        ranks = comm.Get_size()
        res += rank, ranks
    return res


@assert_mpi
def get_local_comm(comm, return_ranks=False, host=None, node_rank=None):
    """Get local comm.

    Split ``comm`` by host. The returned comm is an MPI Comm object that connects ranks from the same host.

    Args:
        comm: MPI comm.
        return_ranks: Whether to return local rank and the number of local ranks as well.
        host: Host name. Will be determined if not provided.
        node_rank: Node rank. Will be determined if not provided.

    Returns:
        Local MPI Comm.
    """
    rank = comm.Get_rank()
    if None in (host, node_rank):
        host, _, node_rank, _ = get_hosts(comm, return_ranks=True)
    comm_local = comm.Split(color=node_rank, key=rank)
    res = comm_local,
    if return_ranks:
        local_rank = comm_local.Get_rank()
        local_ranks = comm_local.Get_size()
        res += (local_rank, local_ranks)
    return res


@assert_mpi
def all_filter(comm, condition: bool):
    """All filter.

    Filter ranks by condition.

    Args:
        comm: MPI Comm.
        condition: Boolean condition.

    Returns:
        Tuple with sets of ranks: (ranks that met the condition, ranks that did not meet the condition)
    """
    keep = set(k for k, v in comm.allgather((comm.Get_rank(), condition)) if v)
    return keep, set(range(comm.Get_size())) - keep


@assert_mpi
def recv(comm, source=ANY_SOURCE, tag=ANY_TAG, status=..., **kwargs):
    """Receive.

    Receive item from ``source`` rank.

    Args:
        comm: MPI Comm
        source: Source rank. Default: any
        tag: Tag. Default: any
        status: Optional ``MPI.Status`` object.
        **kwargs: Keyword arguments for comm.recv.

    Returns:
        Received item and ``MPI.Status`` object.
    """
    if status is ...:
        status = MPI.Status()
    return comm.recv(source=source, tag=tag, status=status, **kwargs), status


@assert_mpi
def send(comm, item, dest, tag=0, **kwargs):
    """Send.

    Send ``item`` via MPI Comm ``comm`` to destination ``dest``.

    Args:
        comm: MPI Comm.
        item: Item.
        dest: Destination rank or `MPI.Status` object.
        tag: Tag.
        **kwargs: Keyword arguments for comm.send.

    """
    if isinstance(dest, MPI.Status):
        dest = dest.Get_source()
    comm.send(item, dest=dest, tag=tag, **kwargs)


def ensure_set(v):
    if isinstance(v, set):
        pass
    elif isinstance(v, int):
        v = {v}
    elif isinstance(v, (list, tuple)):
        v = set(v)
    else:
        raise ValueError(f'Could not handle data type: {type(v)}')
    return v


@assert_mpi
def sink(comm, ranks: set):
    """Sink generator.

    Receive items from `ranks` until receiving `StopIteration`.

    Examples:
        >>> from celldetection import mpi
        ... worker_ranks = {1, 2, 3}
        ... for idx, item in mpi.sink(comm, ranks=worker_ranks):  # receive items from worker_ranks
        ...     pass  # handle item

    Args:
        comm: MPI Comm.
        ranks: Source ranks. All sources have to report `StopIteration` to close sink.

    """
    ranks = ensure_set(ranks)
    while len(ranks) > 0:
        item, status = recv(comm)
        if not (isinstance(item, StopIteration) or item is StopIteration):
            yield status.Get_tag(), item
        else:
            ranks -= {status.Get_source()}


@assert_mpi
def query(comm, source: int, forward_stop_signal=None):
    """Query generator.

    Query items from `source` (serving rank) until receiving `StopIteration`.

    Examples:
        >>> from celldetection import mpi
        >>> server_rank = 0
        >>> for idx, item in mpi.query(comm, server_rank):
        ...     result = process(item)  # handle item
        ...     mpi.send(comm, result, server_rank, tag=idx)  # send result to server

        >>> server_rank = 0
        >>> sink_rank = 1
        >>> for idx, item in mpi.query(comm, server_rank, sink_rank):
        ...     result = process(item)  # handle item
        ...     mpi.send(comm, result, sink_rank, tag=idx)  # send result to sink

    Args:
        comm: MPI Comm.
        source: Source rank.
        forward_stop_signal: Optional ranks that receive a `StopIteration` signal if query is terminated.

    """
    while True:
        comm.send(next, dest=source, tag=0)
        item, status = recv(comm, source)
        if not (isinstance(item, StopIteration) or item is StopIteration):
            yield status.Get_tag(), item
        else:
            break
    if forward_stop_signal is not None:
        fss = forward_stop_signal
        fss = fss if isinstance(fss, (list, tuple)) else [fss]
        for dst in fss:
            send(comm, StopIteration, dst)


@assert_mpi
def serve(comm, ranks: set, iterator, progress=False, desc=None, stats=None):
    """Serve.

    Serves items of `iterator` to `ranks`.
    Once all items have been served, `ranks` receive `StopIteration`.

    Examples:
        >>> from celldetection import mpi
        ... mpi.serve(comm, iterator=range(10), ranks={1, 2, 3})  # serve iterator to ranks

    Args:
        comm: MPI Comm.
        ranks: Client ranks.
        iterator: Iterator.
        progress: Whether to show progress.
        desc: Description, visible in progress report.
        stats: Dictionary of callbacks: {stat_name: callback}

    Returns:
        List of results if `ranks` send results, None otherwise.
        Results are sorted by received tags.
    """
    ranks = ensure_set(ranks)
    results = []
    indices = []
    enum = enumerate(iterator)
    if progress:
        from tqdm import tqdm
        enum = tqdm(enum, total=len(iterator), desc=str(desc))
    for idx, item in enum:
        result, status = recv(comm)
        if not (isinstance(result, type(next)) or result is next):
            indices.append(status.Get_tag())
            results.append(result)
        send(comm, item, status, tag=idx)
        if progress and stats is not None:
            enum.desc = ' - '.join([desc] + [str(v()) for v in stats])
    while len(ranks):
        result, status = recv(comm)
        ranks -= {status.Get_source()}
        if not (isinstance(result, type(next)) or result is next):
            indices.append(status.Get_tag())
            results.append(result)
        send(comm, StopIteration, status)
    assert len(ranks) == 0
    if len(results) > 0:
        results = [results[i] for i in np.argsort(indices)]
    else:
        results = None
    return results
