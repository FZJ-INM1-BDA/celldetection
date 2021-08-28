import numpy as np

_ERR = None
try:
    from mpi4py import MPI
except ModuleNotFoundError as e:
    _ERR = e
    MPI = False

__all__ = ['recv', 'send', 'sink', 'query', 'serve']
ANY_TAG = -1
ANY_SOURCE = -2


def assert_mpi(func):
    def func_wrapper(self, *a, **k):
        if not MPI:
            raise ModuleNotFoundError(
                f'In order to use mpi functions, MPI must be installed. Could not import mpi4py.\n\n'
                f'Check out: https://mpi4py.readthedocs.io/en/stable/install.html\n\n{str(_ERR)}')
        return func(self, *a, **k)

    return func_wrapper


@assert_mpi
def recv(comm, source=ANY_SOURCE, tag=ANY_TAG, status=..., **kwargs):
    if status is ...:
        status = MPI.Status()
    return comm.recv(source=source, tag=tag, status=status, **kwargs), status


@assert_mpi
def send(comm, item, dest, tag=0, **kwargs):
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
        ```
        >>> worker_ranks = {1, 2, 3}
        >>> for idx, item in sink(comm, ranks=worker_ranks):
        ...     pass  # handle item
        ```

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
        ```
        >>> server_rank = 0
        >>> for idx, item in query(comm, server_rank):
        ...     result = process(item)  # handle item
        ...     send(comm, result, server_rank, tag=idx)  # send result to server
        ```
        ```
        >>> server_rank = 0
        >>> sink_rank = 1
        >>> for idx, item in query(comm, server_rank, sink_rank):
        ...     result = process(item)  # handle item
        ...     send(comm, result, sink_rank, tag=idx)  # send result to sink
        ```

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
    for _ in range(len(ranks)):
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
