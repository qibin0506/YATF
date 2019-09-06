import numpy as np

_default_graph = None


class Operation:

    def __init__(self, input_nodes=[]):
        self.input_nodes = input_nodes
        self.consumers = []

        self.inputs = None
        self.output = None

        for input_node in self.input_nodes:
            input_node.consumers.append(self)

        _default_graph.consumers.append(self)

    def compute(self):
        pass


class Add(Operation):

    def __init__(self, x, y):
        super().__init__([x, y])

    def compute(self):
        self.output = self._add(*self.inputs)
        return self.output

    def _add(self, x_value, y_value):
        return x_value + y_value


class Matmul(Operation):

    def __init__(self, x, y):
        super().__init__([x, y])

    def compute(self):
        self.output = self._mul(*self.inputs)
        return self.output

    def _mul(self, x_value, y_value):
        return x_value.dot(y_value)


class Placeholder:

    def __init__(self):
        self.consumers = []
        _default_graph.placeholders.append(self)


class Variable:

    def __init__(self, init_value=None):
        self.value = init_value
        self.consumers = []
        _default_graph.variables.append(self)


class Graph:

    def __init__(self):
        self.consumers = []
        self.placeholders = []
        self.variables = []

    def as_default(self):
        global _default_graph
        _default_graph = self


# test
Graph().as_default()

A = Variable([[1, 0], [0, -1]])
b = Variable([1, 1])

x = Placeholder()

y = Matmul(A, x)
z = Add(y, b)


# end test


class Session:

    def run(self, operations, feed_dict={}):
        outputs = []
        for operation in operations:
            nodes_postorder = self._traverse_postorder(operation)

            for node in nodes_postorder:
                if type(node) == Placeholder:
                    node.output = feed_dict[node]
                elif type(node) == Variable:
                    node.output = node.value
                else:
                    node.inputs = [input_node.output for input_node in node.input_nodes]
                    node.compute()

                if type(node.output) == list:
                    node.output = np.array(node.output)

            outputs.append(operation.output)

        return outputs

    def _traverse_postorder(self, operation):
        nodes_postorder = []

        def recurse(node):
            if isinstance(node, Operation):
                for input_node in node.input_nodes:
                    recurse(input_node)
            nodes_postorder.append(node)

        recurse(operation)

        return nodes_postorder


# test
sess = Session()
output = sess.run([z], feed_dict={x: [1, 2]})


# print(output)
# end test


class Sigmoid(Operation):

    def __init__(self, a):
        super().__init__([a])

    def compute(self):
        self.output = self._sigmoid(*self.inputs)
        return self.output

    def _sigmoid(self, a_value):
        return 1 / (1 + np.exp(-a_value))


# test
Graph().as_default()

x = Placeholder()
w = Variable([1, 1])
b = Variable(0)

p = Sigmoid(Add(Matmul(w, x), b))

sess = Session()
output = sess.run([p], feed_dict={x: [3, 2]})


# print(output)
# end test


class Softmax(Operation):

    def __init__(self, a):
        super().__init__([a])

    def compute(self):
        self.output = self._softmax(*self.inputs)
        return self.output

    def _softmax(self, a_value):
        return np.exp(a_value) / np.sum(np.exp(a_value), axis=1)[:, None]


# test
Graph().as_default()

x = Placeholder()
w = Variable([[1, -1], [1, -1]])
b = Variable([0, 0])

p = Sigmoid(Add(Matmul(w, x), b))

sess = Session()
output = sess.run([p], feed_dict={x: [2, 4]})


# print(output)
# end test


class Log(Operation):

    def __init__(self, x):
        super().__init__([x])

    def compute(self):
        self.output = self._log(*self.inputs)
        return self.output

    def _log(self, x_value):
        return np.log(x_value)


class Multiply(Operation):

    def __init__(self, x, y):
        super().__init__([x, y])

    def compute(self):
        self.output = self._multiply(*self.inputs)
        return self.output

    def _multiply(self, x_value, y_value):
        return x_value * y_value


class ReduceSum(Operation):

    def __init__(self, A, axis=None):
        super().__init__([A])
        self.axis = axis

    def compute(self):
        self.output = self._reduce_sum(*self.inputs)
        return self.output

    def _reduce_sum(self, A_value):
        return np.sum(A_value, axis=self.axis)


class Negative(Operation):

    def __init__(self, x):
        super().__init__([x])

    def compute(self):
        self.output = self._negative(*self.inputs)
        return self.output

    def _negative(self, x_value):
        return -x_value


# test
# Create red points centered at (-2, -2)
red_points = np.random.randn(50, 2) - 2 * np.ones((50, 2))

# Create blue points centered at (2, 2)
blue_points = np.random.randn(50, 2) + 2 * np.ones((50, 2))

Graph().as_default()

X = Placeholder()
c = Placeholder()

W = Variable(
    [
        [1, -1],
        [1, -1]
    ]
)

b = Variable([0, 0])

p = Softmax(Add(Matmul(X, W), b))
# cross-entropy loss
J = Negative(ReduceSum(ReduceSum(Multiply(c, Log(p)), axis=1)))

sess = Session()
output = sess.run([J], feed_dict={
    X: np.concatenate((blue_points, red_points)),
    c: [[1, 0]] * len(blue_points) + [[0, 1]] * len(red_points)
})
# print(output)
# end test

from queue import Queue


class GradientDescentOptimizer:

    def __init__(self, lr):
        self.lr = lr

    def minimize(self, loss):
        lr = self.lr

        class MinizationOperation(Operation):
            def compute(self):
                grad_table = compute_gradients(loss)
                for node in grad_table:
                    if type(node) == Variable:
                        grad = grad_table[node]
                        node.value -= lr * grad

        return MinizationOperation()


_gradient_registry = {}


class RegisterGradient:

    def __init__(self, op_type):
        self.op_type = eval(op_type)

    def __call__(self, f):
        _gradient_registry[self.op_type] = f
        return f


def compute_gradients(loss):
    grad_table = {}
    grad_table[loss] = 1

    visited = set()
    queue = Queue()

    visited.add(loss)
    queue.put(loss)

    while not queue.empty():
        node = queue.get()

        if node != loss:
            grad_table[node] = 0
            for consumer in node.consumers:
                lossgrad_wrt_consumer_output = grad_table[consumer]
                consumer_op_type = consumer.__class__

                bprop = _gradient_registry[consumer_op_type]
                lossgrads_wrt_consumer_inputs = bprop(consumer, lossgrad_wrt_consumer_output)

                if len(consumer.input_nodes) == 1:
                    grad_table[node] += lossgrads_wrt_consumer_inputs
                else:
                    node_index_in_consumer_inputs = consumer.input_nodes.index(node)
                    lossgrad_wrt_node = lossgrads_wrt_consumer_inputs[node_index_in_consumer_inputs]
                    grad_table[node] += lossgrad_wrt_node

        if hasattr(node, "input_nodes"):
            for input_node in node.input_nodes:
                if input_node not in visited:
                    visited.add(input_node)
                    queue.put(input_node)

    return grad_table


@RegisterGradient("Add")
def _add_gradient(op, grad):
    a = op.inputs[0]
    b = op.inputs[1]

    grad_wrt_a = grad
    while np.ndim(grad_wrt_a) > len(a.shape):
        grad_wrt_a = np.sum(grad_wrt_a, axis=0)
    for axis, size in enumerate(a.shape):
        if size == 1:
            grad_wrt_a = np.sum(grad_wrt_a, axis=axis, keepdims=True)

    grad_wrt_b = grad
    while np.ndim(grad_wrt_b) > len(b.shape):
        grad_wrt_b = np.sum(grad_wrt_b, axis=0)
    for axis, size in enumerate(a.shape):
        if size == 1:
            grad_wrt_b = np.sum(grad_wrt_b, axis=axis, keepdims=True)

    return [grad_wrt_a, grad_wrt_b]


@RegisterGradient("Matmul")
def _matmul_gradient(op, grad):
    A = op.inputs[0]
    B = op.inputs[1]

    return [grad.dot(B.T), A.T.dot(grad)]


@RegisterGradient("Sigmoid")
def _sigmoid_gradient(op, grad):
    sigmoid = op.output
    return grad * sigmoid * (1 - sigmoid)


@RegisterGradient("Softmax")
def _softmax_gradient(op, grad):
    softmax = op.output

    return (grad - np.reshape(np.sum(grad * softmax, 1), [-1, 1])) * softmax


@RegisterGradient("Log")
def _log_gradient(op, grad):
    x = op.inputs[0]
    return grad / x


@RegisterGradient("Multiply")
def _multipy_gradient(op, grad):
    A = op.inputs[0]
    B = op.inputs[1]

    return [grad * B, grad * A]


@RegisterGradient("ReduceSum")
def _reduce_sum_gradient(op, grad):
    A = op.inputs[0]

    output_shape = np.array(A.shape)
    output_shape[op.axis] = 1
    tile_scaling = A.shape // output_shape
    grad = np.reshape(grad, output_shape)

    return np.tile(grad, tile_scaling)


@RegisterGradient("Negative")
def _negative_gradient(op, grad):
    return -grad

# test
print("train start...")

Graph().as_default()

X = Placeholder()
c = Placeholder()

W = Variable(np.random.randn(2, 2))
b = Variable(np.random.randn(2))

p = Softmax(Add(Matmul(X, W), b))

loss = Negative(ReduceSum(ReduceSum(Multiply(c, Log(p)), axis=1)))

train_op = GradientDescentOptimizer(lr=0.01).minimize(loss)

sess = Session()

# initialized
sess.run([loss], feed_dict={
    X: np.concatenate((blue_points, red_points)),
    c: [[1, 0]] * len(blue_points) + [[0, 1]] * len(red_points)
})

for step in range(1000):
    _, loss_value = sess.run([train_op, loss], feed_dict={
        X: np.concatenate((blue_points, red_points)),
        c: [[1, 0]] * len(blue_points) + [[0, 1]] * len(red_points)
    })

    print("loss:{}".format(loss_value))

# end test
