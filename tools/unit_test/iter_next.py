class A(object):
    def __init__(self):
        self.array = [1, 2, 3]
        self.index = 0
    def __iter__(self):
        return self
    def __next__(self):
        self.index += 1
        return self.array[self.index]

x = A()
iter
print(next(x))