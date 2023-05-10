from multiprocessing import Pool
import gc
import psutil

global_pool = None
global_pool_results = []
num_processes = 0


def init_pool(processes):
    global global_pool
    global global_pool_results
    global num_processes
    if global_pool is not None:
        raise RuntimeError("Pool was already initialized.")
    global_pool = Pool(processes)
    global_pool_results = []
    num_processes = processes
    return global_pool, global_pool_results


def get_pool(processes=1):
    global global_pool
    global global_pool_results
    if global_pool is None:
        global_pool = Pool(processes)
        global_pool_results = []
    return global_pool, global_pool_results


def get_results(delete=False):
    global global_pool_results
    results = [i.get() for i in global_pool_results]
    global_pool_results = []
    if delete:
        del results
        gc.collect()
        return []
    return results


def queue_job(func, *args, max_mem_usage=0.8):
    global global_pool_results
    global num_processes
    if len(global_pool_results) >= num_processes or float(get_memory_usage()) / 100 > max_mem_usage:
        get_results()
    global_pool_results.append(func(*args))


def close_pool():
    global global_pool
    if global_pool is not None:
        get_results()
        global_pool.close()
        global_pool.join()


def get_memory_usage():
    return psutil.virtual_memory()[2]


