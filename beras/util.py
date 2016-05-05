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

from keras.engine.topology import merge, InputLayer
from keras.engine.training import collect_trainable_weights
import theano.tensor as T
import copy
import re


def add_border(input, border, mode='repeat'):
    if mode == 'repeat':
        return add_border_repeat(input, border), 'valid'
    elif mode == 'zero':
        if hasattr(border, 'eval'):
            border = int(border.eval())
        return input, border
    else:
        raise ValueError("Invalid mode: {}".format(mode))


def add_border_repeat(input, border):
    if type(border) is int:
        border = (border,) * input.ndim

    w_start = input[:, :, :, :1]
    w_start_padding = T.repeat(w_start, border, axis=3)
    w_end = input[:, :, :, -1:]
    w_end_padding = T.repeat(w_end, border, axis=3)

    w_padded = T.concatenate([w_start_padding, input,
                              w_end_padding], axis=3)

    h_start = w_padded[:, :, :1, :]
    h_start_padding = T.repeat(h_start, border, axis=2)
    h_end = w_padded[:, :, -1:, :]
    h_end_padding = T.repeat(h_end, border, axis=2)

    padded = T.concatenate([h_start_padding, w_padded,
                            h_end_padding], axis=2)
    return padded


def sequential(layers, ns=None, trainable=True):
    def flatten(xs):
        for x in xs:
            try:
                for f in flatten(x):
                    yield f
            except TypeError:
                yield x

    for i, l in enumerate(flatten(layers)):
        l.name.split('_')
        if ns is not None:
            if '.' not in l.name:
                name = re.sub('_\d+$', '', l.name)
                name = "{:02}_{}".format(i, name)
            l.name = ns + '.' + name
        l.trainable = trainable

    def call(input):
        x = input
        for l in flatten(layers):
            x = l(x)
        return x

    return call


def concat(tensors, axis=1, name=None, output_shape=None):
    if type(tensors) not in (list, tuple):
        return tensors
    elif len(tensors) == 1:
        return tensors[0]

    return merge(tensors, mode='concat', concat_axis=axis,
                 name=name, output_shape=output_shape)


def namespace(namespace, inputs, outputs):
    layers = collect_layers(inputs, outputs)
    for l in layers:
        l.name = namespace + '.' + l.name


def rename_layer(keras_tensor, name):
    layer = keras_tensor._keras_history[0]
    layer.name = name


def collect_layers(inputs, outputs):
    if type(inputs) not in (tuple, list):
        inputs = [inputs]

    if type(outputs) not in (tuple, list):
        outputs = [outputs]

    # container_nodes: set of nodes included in the graph
    # (not all nodes included in the layers are relevant to the current graph).
    container_nodes = set()  # ids of all nodes relevant to the Container
    nodes_depths = {}  # map {node: depth value}
    layers_depths = {}  # map {layer: depth value}

    def build_map_of_graph(tensor, seen_nodes=set(), depth=0,
                           layer=None, node_index=None, tensor_index=None):
        '''This recursively updates the maps nodes_depths,
        layers_depths and the set container_nodes.
        Does not try to detect cycles in graph (TODO?)

        # Arguments
            tensor: some tensor in a graph
            seen_nodes: set of node ids ("{layer.name}_ib-{node_index}")
                of nodes seen so far. Useful to prevent infinite loops.
            depth: current depth in the graph (0 = last output).
            layer: layer from which `tensor` comes from. If not provided,
                will be obtained from `tensor._keras_history`.
            node_index: node index from which `tensor` comes from.
            tensor_index: tensor_index from which `tensor` comes from.
        '''
        if not layer or node_index is None or tensor_index is None:
            layer, node_index, tensor_index = tensor._keras_history
        node = layer.inbound_nodes[node_index]

        # prevent cycles
        if node in seen_nodes:
            return
        seen_nodes.add(node)

        # basic sanity checks
        assert node.outbound_layer == layer
        assert node.output_tensors[tensor_index] == tensor

        node_key = layer.name + '_ib-' + str(node_index)
        # update container_nodes
        container_nodes.add(node_key)
        # update nodes_depths
        if node not in nodes_depths:
            nodes_depths[node] = depth
        else:
            nodes_depths[node] = max(depth, nodes_depths[node])
        # update layers_depths
        if layer not in layers_depths:
            layers_depths[layer] = depth
        else:
            layers_depths[layer] = max(depth, layers_depths[layer])

        # propagate to all previous tensors connected to this node
        for i in range(len(node.inbound_layers)):
            x = node.input_tensors[i]
            layer = node.inbound_layers[i]
            if x in inputs or layer in inputs:
                continue
            node_index = node.node_indices[i]
            tensor_index = node.tensor_indices[i]
            build_map_of_graph(x, copy.copy(seen_nodes), depth + 1,
                               layer, node_index, tensor_index)

    for x in outputs:
        build_map_of_graph(x, seen_nodes=set(), depth=0)

    # build a map {depth: list of nodes with this depth}
    nodes_by_depth = {}
    for node, depth in nodes_depths.items():
        if depth not in nodes_by_depth:
            nodes_by_depth[depth] = []
        nodes_by_depth[depth].append(node)

    # build a map {depth: list of layers with this depth}
    layers_by_depth = {}
    for layer, depth in layers_depths.items():
        if depth not in layers_by_depth:
            layers_by_depth[depth] = []
        layers_by_depth[depth].append(layer)

    depth_keys = list(nodes_by_depth.keys())
    depth_keys.sort(reverse=True)

    # set layers and layers_by_depth
    layers = []
    for depth in depth_keys:
        layers_for_depth = layers_by_depth[depth]
        # container.layers needs to have a deterministic order
        layers_for_depth.sort(key=lambda x: x.name)
        for layer in layers_for_depth:
            layers.append(layer)
    layers = layers
    layers_by_depth = layers_by_depth

    computable_tensors = []
    for x in inputs:
        computable_tensors.append(x)

    layers_with_complete_input = []  # to provide a better error msg
    for depth in depth_keys:
        for node in nodes_by_depth[depth]:
            layer = node.outbound_layer
            if layer:
                for x in node.input_tensors:
                    if x not in computable_tensors:
                        raise Exception(
                            'Graph disconnected: '
                            'cannot obtain value for tensor ' +
                            str(x) + ' at layer "' + layer.name + '". '
                            'The following previous layers '
                            'were accessed without issue: ' +
                            str(layers_with_complete_input))
                for x in node.output_tensors:
                    computable_tensors.append(x)
                layers_with_complete_input.append(layer.name)

    # set nodes and nodes_by_depth
    container_nodes = container_nodes
    nodes_by_depth = nodes_by_depth
    return layers


def trainable_weights(layers):
    weights = []
    for layer in layers:
        weights += collect_trainable_weights(layer)
    return weights

