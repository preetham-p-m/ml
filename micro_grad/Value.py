import math


class Value:
    def __init__(self, data, _children=(), op="", label="") -> None:
        self.data = data
        self._prev = set(_children)  # What all objects are involved to form this Object
        self._op = op  # What operation was performed to get this data
        self.label = label
        self.grad: float = 0
        self._backward = lambda: None

    def __repr__(self) -> str:
        return f"Value(data={self.data})"

    def __add__(self, other):
        other = self.__type_validate_and_convert(other)
        out = Value(self.data + other.data, (self, other), "+")

        def backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad

        out._backward = backward

        return out

    def __mul__(self, other):
        other = self.__type_validate_and_convert(other)
        out = Value(self.data * other.data, (self, other), "*")

        def backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = backward
        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), (
            "only supported int/float powers for now"
        )
        out = Value(
            self.data**other,
            (self,),
            f"**{other}",
        )

        def backward():
            self.grad += (other * self.data ** (other - 1)) * out.grad

        out._backward = backward

        return out

    def tanh(self):
        x = self.data
        t = (math.exp(2 * x) - 1) / (math.exp(2 * x) + 1)
        out = Value(t, (self,), "tanh")

        def backward():
            self.grad += (1 - t**2) * out.grad

        out._backward = backward
        return out

    def exp(self):
        x = self.data
        t = math.exp(x)
        out = Value(t, (self,), "exp")

        def backward():
            self.grad += t * out.grad

        out._backward = backward
        return out

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self + (-other)

    def __neg__(self):
        return self * -1

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        other = self.__type_validate_and_convert(other)
        return self * (other**-1)

    def __rtruediv__(self, other):
        other = self.__type_validate_and_convert(other)
        return other / self

    def backward(self):
        # build topological ordering
        toposort = []
        visited = set()

        def build_topological_sort(node: Value):
            if node in visited:
                return
            visited.add(node)
            for ch in node._prev:
                build_topological_sort(ch)
            toposort.append(node)

        build_topological_sort(self)

        # seed gradient on output
        self.grad = 1.0

        # run backward in reverse topological order
        for node in reversed(toposort):
            node._backward()

    @staticmethod
    def __type_validate_and_convert(other):
        return other if isinstance(other, Value) else Value(other)
