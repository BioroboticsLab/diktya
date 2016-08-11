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

from __future__ import absolute_import
from __future__ import print_function
from keras.engine.training import make_batches, slice_X
import numpy as np

from keras import callbacks as cbks
from keras.utils.generic_utils import Progbar


class AbstractModel(object):
    @staticmethod
    def _train_fn(model, f, ins_batch, batch_logs):
        outs = f(*ins_batch)
        if type(outs) != list:
            outs = [outs]
        return outs

    @staticmethod
    def _val_fn(model, val_f, batch_size, val_ins, batch_logs):
        # replace with self._evaluate
        val_outs = model._test_loop(val_f, val_ins, batch_size=batch_size, verbose=0)
        if type(val_outs) != list:
            val_outs = [val_outs]
        return val_outs

    def _fit(self, f, nb_train_sample, nb_batches, batch_size=128, nb_epoch=100,
             verbose=1, callbacks=[], shuffle=True, metrics=[]):
        """
            Abstract fit function for f(*ins). Assume that f returns a list,
            labelled by out_labels.  """

        history = cbks.History()
        callbacks = [cbks.BaseLogger()] + callbacks + [history]
        if verbose:
            callbacks = callbacks + [cbks.ProgbarLogger()]

        callbacks = cbks.CallbackList(callbacks)
        callbacks._set_model(self)
        callbacks._set_params({
            'batch_size': nb_train_sample // nb_batches,
            'nb_epoch': nb_epoch,
            'nb_sample': nb_train_sample,
            'verbose': verbose,
            'do_validation': False,
            'metrics': metrics,
        })
        callbacks.on_train_begin()

        self.stop_training = False
        for epoch in range(nb_epoch):
            callbacks.on_epoch_begin(epoch)
            for batch_index in range(nb_batches):
                batch_logs = {}
                batch_logs['batch'] = batch_index
                batch_logs['size'] = batch_size
                callbacks.on_batch_begin(batch_index, batch_logs)

                f(self, batch_index, batch_logs)
                callbacks.on_batch_end(batch_index, batch_logs)
                epoch_logs = {}

            callbacks.on_epoch_end(epoch, epoch_logs)
            if self.stop_training:
                break

        callbacks.on_train_end()
        return history

    def _predict_loop(self, f, ins, batch_size=128, verbose=0):
        '''
            Abstract method to loop over some data in batches.
        '''
        nb_sample = len(ins[0])
        outs = []
        if verbose == 1:
            progbar = Progbar(target=nb_sample)
        batches = make_batches(nb_sample, batch_size)
        index_array = np.arange(nb_sample)
        for batch_index, (batch_start, batch_end) in enumerate(batches):
            batch_ids = index_array[batch_start:batch_end]
            ins_batch = slice_X(ins, batch_ids)
            batch_outs = f(self, ins_batch)
            if type(batch_outs) != list:
                batch_outs = [batch_outs]
            if batch_index == 0:
                for batch_out in batch_outs:
                    shape = (nb_sample,) + batch_out.shape[1:]
                    outs.append(np.zeros(shape))

            for i, batch_out in enumerate(batch_outs):
                outs[i][batch_start:batch_end] = batch_out
            if verbose == 1:
                progbar.update(batch_end)
        return outs

    def _test_loop(self, f, ins, batch_size=128, verbose=0):
        '''
            Abstract method to loop over some data in batches.
        '''
        nb_sample = len(ins[0])
        outs = []
        if verbose == 1:
            progbar = Progbar(target=nb_sample)
        batches = make_batches(nb_sample, batch_size)
        index_array = np.arange(nb_sample)
        for batch_index, (batch_start, batch_end) in enumerate(batches):
            batch_ids = index_array[batch_start:batch_end]
            ins_batch = slice_X(ins, batch_ids)

            batch_outs = f(self, ins_batch)
            if type(batch_outs) == list:
                if batch_index == 0:
                    for batch_out in enumerate(batch_outs):
                        outs.append(0.)
                for i, batch_out in enumerate(batch_outs):
                    outs[i] += batch_out * len(batch_ids)
            else:
                if batch_index == 0:
                    outs.append(0.)
                outs[0] += batch_outs * len(batch_ids)

            if verbose == 1:
                progbar.update(batch_end)
        for i, out in enumerate(outs):
            outs[i] /= nb_sample
        return outs

    def get_config(self, verbose=0):
        raise NotImplemented()

    def to_yaml(self):
        # dump model configuration to yaml string
        import yaml
        config = self.get_config()
        return yaml.dump(config)

    def to_json(self):
        # dump model configuration to json string
        import json
        config = self.get_config()
        return json.dumps(config)
