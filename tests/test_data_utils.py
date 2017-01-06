from diktya.data_utils import multiprocessing_generator


def test_multiprocessing_generator():
    dummy_range = list(range(10))

    def make_dummy_generator():
        for value in dummy_range:
            yield value

    generator = multiprocessing_generator(make_dummy_generator, num_processes=2)

    results = list(generator)
    assert(len(results) == 2 * len(dummy_range))
    assert(set(results) == set(dummy_range))
