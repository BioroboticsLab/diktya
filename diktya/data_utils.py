# Copyright 2015 Leon Sixt
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
from collections import defaultdict
from multiprocessing import Event, Process, Queue
import h5py


class HDF5Tensor:
    def __init__(self, datapath, dataset, start, end, normalizer=None):
        self.refs = defaultdict(int)
        if datapath not in list(self.refs.keys()):
            f = h5py.File(datapath)
            self.refs[datapath] = f
        else:
            f = self.refs[datapath]
        self.start = start
        self.end = end
        self.data = f[dataset]
        self.normalizer = normalizer

    def __len__(self):
        return self.end - self.start

    def __getitem__(self, key):
        if isinstance(key, slice):
            if key.stop + self.start <= self.end:
                idx = slice(key.start+self.start, key.stop + self.start)
            else:
                raise IndexError
        elif isinstance(key, int):
            if key + self.start < self.end:
                idx = key+self.start
            else:
                raise IndexError
        elif isinstance(key, np.ndarray):
            if np.max(key) + self.start < self.end:
                idx = (self.start + key).tolist()
            else:
                raise IndexError
        elif isinstance(key, list):
            if max(key) + self.start < self.end:
                idx = [x + self.start for x in key]
            else:
                raise IndexError
        return self.data[idx]

    @property
    def shape(self):
        return tuple((self.end - self.start, ) + self.data.shape[2:])


def probabilistic_mean(tensor, nb_samples=10000, axis=0):
    nb_samples = min(nb_samples, len(tensor))
    idx = np.sort(np.random.choice(len(tensor), nb_samples, replace=False))
    samples = tensor[idx]
    return np.mean(samples, axis=axis, dtype=np.float64)


def multiprocessing_generator(generator_factory, num_processes=1):
    def generator_process(generator_factory, batch_queue, exit):
        np.random.seed()
        gen = generator_factory()

        for value in gen:
            queue.put(value)

        exit.set()

    queue = Queue(num_processes * 2)

    exit_events = []
    processes = []
    for _ in range(num_processes):
        exit_event = Event()
        process = Process(target=generator_process, args=(generator_factory, queue, exit_event))
        process.start()
        exit_events.append(exit_event)
        processes.append(process)

    try:
        while any([not e.is_set() for e in exit_events]) or not queue.empty():
            yield queue.get()
    except Exception as e:
        print(e)

    for process in processes:
        process.join()
