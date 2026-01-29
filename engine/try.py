import sys
from itertools import chain
d = {3: ["residual", "shortcut"]}
import tracemalloc
tracemalloc.start()
a = {3: ["residual", "shortcut"]}
current, peak = tracemalloc.get_traced_memory()
tracemalloc.stop()
print(current, peak)