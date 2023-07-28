ZTF sensors
===========

Code to characterize, model and correct the main effects observed on the ZTF sensors:
  - linearity
  - brighter-fatter effect
  - *pocket effect*, a.k.a. *ccd 6 effect* 


Installation
------------

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

Building the documentation
--------------------------

```bash
cd zfsensors/docs
make html
```


