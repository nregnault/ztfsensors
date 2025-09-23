import sys
import timeit
from statistics import median, stdev

import numpy as np
from ztfsensors import test
from ztfsensors.pocket import pocket_model_derivatives  # noqa

model, pixels = test.get_pocket_test()
test_column = np.full(pixels.shape[0], np.median(pixels))

# %timeit pocket_model.apply(pixels.copy(), backend="numpy-nr");
# %timeit pocket_model.apply(pixels.copy(), backend="numpy");
# %timeit pocket_model.apply(pixels.copy(), backend="jax");
# %timeit pocket_model.apply(pixels.copy(), backend="cpp");

backends = ["numpy-nr", "numpy", "cpp"]  # "jax",
number, repeat = 5, 5

benchmarks = {
    "apply": 'model.apply(pixels.copy(), backend="{}")',
    "onecol": 'model.apply(pixels[1].copy(), backend="{}")',
    "deriv": 'pocket_model_derivatives(model, test_column, backend="{}")',
}

if len(sys.argv) != 2:
    sys.exit(f"usage: python {sys.argv[0]} [apply|onecol|deriv|all]")


if sys.argv[1] == "all":
    names = list(benchmarks.keys())
else:
    names = [sys.argv[1]]
    assert names[0] in benchmarks.keys()

for name in names:
    code = benchmarks[name]
    print(f"running benchmark {name}")
    for backend in backends:
        print(f"{backend:10s}: ", end="", flush=True)
        res = timeit.repeat(
            code.format(backend), globals=globals(), number=number, repeat=repeat
        )
        res = [x / number * 1000 for x in res]
        if len(res) > 2:
            res = res[1:]
        print(f"{median(res[1:]):.1f}±{stdev(res):.1f}ms")
    print()
