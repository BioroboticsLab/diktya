
import h5py
import numpy as np

from diktya.hdf5 import create_hdf5_for, append_to_hdf5, shuffle_hdf5, iterate_hdf5


def test_hdf5(tmpdir):
    bs = 10

    def get_data(i):
        np.random.seed(i)
        return {
            'idx': np.arange(i, i+bs, dtype='int'),
            'random': np.random.uniform(-1, 1, (bs, 4, 4)),
            'complex': np.random.normal(-1, 1, (bs, 8)).astype(np.complex128),
        }

    fname = str(tmpdir.join('test.hdf5'))
    h5 = create_hdf5_for(fname, get_data(0))

    append_to_hdf5(h5, get_data(0))
    assert h5.attrs['__append_pos'] == bs

    append_to_hdf5(h5, get_data(1))
    assert h5.attrs['__append_pos'] == 2*bs

    append_to_hdf5(h5, get_data(2))
    assert h5.attrs['__append_pos'] == 3*bs

    for i, batch in enumerate(iterate_hdf5(h5, bs, nb_iterations=1)):
        data = get_data(i)
        assert set(data.keys()) == set(batch.keys())
        for name in data.keys():
            assert (batch[name] == data[name]).all()
    assert i == 2

    for i, batch in enumerate(iterate_hdf5(h5, bs, start=2*bs, nb_iterations=1)):
        pass
    assert i == 0

    for i, batch in enumerate(iterate_hdf5(h5, bs, start=0.50, nb_iterations=1)):
        pass
    assert i == 1

    for i, batch in enumerate(iterate_hdf5(h5, bs, nb_iterations=1, shuffle=True)):
        data = get_data(i)
        assert set(data.keys()) == set(batch.keys())
        for name in data.keys():
            assert (batch[name] != data[name]).any()
    assert i == 2

    shuffled_fname = str(tmpdir.join('test_shuffled.hdf5'))
    shuffle_hdf5(h5, shuffled_fname, print_progress=True)
    h5_shuffled = h5py.File(shuffled_fname, 'r')
    print(np.array(h5_shuffled['idx']))
    assert h5_shuffled['idx'].shape == h5['idx'].shape
    for i, batch in enumerate(iterate_hdf5(h5_shuffled, bs, nb_iterations=1, shuffle=False)):
        data = get_data(i)
        assert set(data.keys()) == set(batch.keys())
        for name in data.keys():
            assert (batch[name] != data[name]).any()
    assert i == 2
