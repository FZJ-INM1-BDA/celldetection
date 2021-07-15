import torch
import gc
from time import time
import numpy as np

__all__ = ['start_timer', 'stop_timer', 'print_timing']

TIMINGS = {}


def convert_seconds(seconds):
    if seconds // 60:
        minutes = seconds // 60
        seconds %= 60
    else:
        minutes = 0
    if minutes // 60:
        hours = minutes // 60
        minutes %= 60
    else:
        hours = 0
    if hours // 24:
        days = hours // 24
        hours %= 24
    else:
        days = 0
    return int(days), int(hours), int(minutes), int(np.round(seconds))


def seconds_to_str(seconds):
    s = [f'{i} {n[:-1] if i == 1 else n}' for i, n in
         zip(convert_seconds(seconds), ('days', 'hours', 'minutes', 'seconds'))]
    s = ', '.join(s)
    return s


def print_timing(name, seconds):
    if seconds >= 1:
        val = np.round(seconds, 3)
        fill = 75 - len(name) - len(str(val))
        print(name + ':', ' ' * fill, val, 's')
        return
    seconds *= 1000
    if seconds >= 1:
        val = np.round(seconds, 3)
        fill = 75 - len(name) - len(str(val))
        print(name + ':', ' ' * fill, val, 'ms')
        return
    seconds *= 1000
    if seconds >= 1:
        val = np.round(seconds, 3)
        fill = 75 - len(name) - len(str(val))
        print(name + ':', ' ' * fill, val, 'µs')
        return
    seconds *= 1000
    val = np.round(seconds, 3)
    fill = 75 - len(name) - len(str(val))
    print(name + ':', ' ' * fill, val, 'ns')


def start_timer(name, cuda=True, collect=True):
    """Keyword PyTorch timer.

    Can be used to measure PyTorch GPU times.

    Args:
        name: Keyword

    Returns:

    """
    global TIMINGS
    if collect:
        gc.collect()
    if cuda:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    TIMINGS[name] = time()


def stop_timer(name, cuda=True, verbose=True):
    global TIMINGS
    if cuda:
        torch.cuda.synchronize()
    delta = time() - TIMINGS[name]
    if verbose:
        print_timing(name, delta)
    return delta
