ZTF sensors
===========

Code to characterize, model and correct the main effects observed on the ZTF
sensors. As of today, we have a tentative correction of the pocket effect (aka
MegaTrap effect, aka CCD6 effect).

Installation
------------

It is recommended to setup a `conda` environment:

``` bash
conda create -n sensors
conda activate sensors

# using mamba instead of conda is much faster
mamba install numpy scipy matplotlib scikit-sparse pybind11 ipython pandas ruamel.yaml
```

Now, you can install ztfsensors:

```bash
# clone this repo 
git clone https://github.com/nregnault/ztfsensors
cd ztfsensors
pip install . 
```

or, if you intend to hack the code:
```
pip install -e . 
```


<!-- Building the documentation -->
<!-- -------------------------- -->

<!-- ```bash -->
<!-- cd zfsensors/docs -->
<!-- make html -->
```


