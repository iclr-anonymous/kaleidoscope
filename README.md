Here we provide fast C++ and CUDA implementations of kaleidoscope matrix multiplication.

## Requirements
Python 3.6+  
PyTorch >=1.2
Numpy

## Usage

* The module `Butterfly` in `butterfly/butterfly.py` can be used as a drop-in
replacement for a `nn.Linear` layer. The files in `butterfly` directory are all
that are needed for this use.

The kaleidoscope multiplication is written in C++ and CUDA as PyTorch extension.
To install it:
```
cd butterfly/factor_multiply
python setup.py install
cd butterfly/factor_multiply_fast
python setup.py install
```
Without the C++/CUDA version installed, kaleidoscope multiplication is still usable, but is
quite slow. The variable `use_extension` in `butterfly/butterfly_multiply.py`
controls whether to use the C++/CUDA version or the pure PyTorch version.
