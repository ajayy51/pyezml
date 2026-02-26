import pandas as pd
class Pipe1:
    def __ror__(self, other):
        return "SUCCESS 1"
class Pipe2:
    __array_priority__ = 10000
    __pandas_priority__ = 10000
    def __ror__(self, other):
        return "SUCCESS 2"

df = pd.DataFrame({"A": [1, 2]})

print("TEST 2:")
try:
    print(df | Pipe2())
except Exception as e:
    print("FAILED 2:", e)

print("TEST 1:")
try:
    print(df | Pipe1())
except Exception as e:
    print("FAILED 1:", e)
