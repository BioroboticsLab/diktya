# Copyright 2016 Leon Sixt
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


import h5py
import numpy as np


def create_hdf5_for(fname, data, chunks=256):
    h5 = h5py.File(fname, 'w')
    h5.attrs['__append_pos'] = 0
    _create_datasets_for(h5, data, chunks)
    return h5


def _create_datasets_for(h5, data, chunks=256):
    for name, array in data.items():
        shape = array.shape[1:]
        h5.create_dataset(
            name,
            shape=(1,) + shape,
            chunks=(chunks,) + shape,
            maxshape=(None,) + shape,
            dtype=str(array.dtype))


def append_to_hdf5(h5, data):
    def ensure_enough_space_for(size):
        for name in data.keys():
            if len(h5[name]) < size:
                h5[name].resize(size, axis=0)

    size = len(next(iter(data.values())))
    begin = int(h5.attrs['__append_pos'])
    end = begin+size

    for name, array in data.items():
            ensure_enough_space_for(end)
            if len(array) != size:
                raise Exception("Arrays must have the same number of samples."
                                " Got {} and {}".format(size, len(array)))
            h5[name][begin:end] = array[:size]
    h5.attrs['__append_pos'] += size
    return int(h5.attrs['__append_pos'])


def shuffle_hdf5(h5, output_fname, batch_size=100, print_progress=False):
    h5_shuffled = create_hdf5_for(
        output_fname, next(iterate_hdf5(h5, 1)))

    for key in h5.attrs.keys():
        h5_shuffled.attrs[key] = h5.attrs[key]

    h5_shuffled.attrs['__append_pos'] = 0

    nb_samples = int(h5.attrs['__append_pos'])
    batch_size = min(batch_size, nb_samples)
    if print_progress:
        from tqdm import tqdm
        bar = tqdm(total=int(nb_samples))

    for i, batch in enumerate(iterate_hdf5(h5, batch_size, shuffle=True, nb_iterations=1)):
        print(i)
        print(next(iter(batch.values())).shape)
        append_to_hdf5(h5_shuffled, batch)
        if print_progress:
            bar.update(batch_size)
    h5_shuffled.close()


def iterate_hdf5(h5, batch_size, shuffle=False,
                 nb_iterations=None, fields=None):
    """
    Iterates over HDF5 file ``h5``.

    Args:
        batch_size (int):
        shuffle (int): randomly shuffle data
        nb_iterations (int): iterate that many times of the data sets. Defaulft is infinit times
        fields (list):  list of field names to yield
    """
    if fields is None:
        fields = list(h5.keys())
    nb_samples = len(h5[fields[0]])
    indicies = np.arange(nb_samples)
    if shuffle:
        np.random.shuffle(indicies)
    iterations_done = 0
    stop_iteration = 0
    i = 0
    while True:
        size = batch_size
        batch = {name: [] for name in fields}
        while size > 0:
            nb = min(nb_samples, i + size) - i
            idx = indicies[i:i + nb]
            idx = np.sort(idx)
            if shuffle:
                shuffle_idx = np.arange(nb)
                np.random.shuffle(shuffle_idx)

            for name in fields:
                arr = h5[name][idx.tolist()]
                if shuffle:
                    arr = arr[shuffle_idx.tolist()]
                batch[name].append(arr)

            size -= nb
            if i + nb >= nb_samples:
                iterations_done += 1
                if iterations_done >= nb_iterations:
                    stop_iteration = True
                    break
            i = (i + nb) % nb_samples
        yield {name: np.concatenate(arrs) for name, arrs in batch.items()}
        if stop_iteration:
            break
