from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup


__version__ = '0.1.3'


# there is some compiled code
ext_modules = [
    Pybind11Extension("ztfsensors._pocket",
                      ["ztfsensors/_pocket.cpp"],
                      define_macros = [('VERSION_INFO', __version__)],
                      ),
    ]

setup(
    name='ztfsensors',
    version=__version__,
    # author="Nicolas Regnault",
    # author_email="nicolas.regnault@lpnhe.in2p3.fr",
    # url="https://github.com/nregnault/ztfsensors",
    # description="models and corrections for the ZTF pocket effect (and BF)",
    # long_description='',
    # packages=['ztfsensors'],
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    # zip_safe=False,
    # python_requires='>=3.7',
    # # install_requires=['saunerie @ git+https://gitlab.in2p3.fr/betoule/saunerie.git']
)
