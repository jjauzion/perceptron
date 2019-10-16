import multiprocessing as mp
import numpy as np
import queue
import os


def pool_worker(val, lock):
    a = np.ones((val[0], val[0]))
    return [np.sum(a), val[0], os.getppid(), val[1]]


def worker(q):
    while True:
        try:
            val = q.get(block=False)
        except queue.Empty:
            print("worker '{}' terminated".format(os.getppid()))
            break
        a = np.ones((val, val))
    return np.sum(a)


if __name__ == "__main__":
    lock = mp.Lock()
    q = mp.Queue()
    l_input = [(4, 1), (7, 2), (3, 3), (8, 4)]
    for v in l_input:
        q.put(v)
    pool = mp.Pool(2)
    res = pool.map(pool_worker, l_input)
    print(np.array(res))
