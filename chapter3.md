# Chapter 3 - It Starts with a Tensor

## Main points:
0. Neural networks transform floating-point representations into other floating-point representations. The starting and ending representations are typically 
human interpretable, but the intermediate representations are less so → These floating-point representations are stored in tensors

1. PyTorch **tensors** (a **multidimensional array**) are much more efficient than Python lists because the entire tensor is a *view over a contiguous block of memory*
and can be defined by its storage, **size**, **offset**, and **stride** (for Python lists, each individual element is a Python object randomly allocated in memory). 
PyTorch tensors (default **dtype** `float32`) can easily be converted to and from **NumPy arrays** (default `float64`)

2. You can copy tensors from **CPU** RAM to **GPU** RAM (and vice versa) using various methods (`to(device='cuda:0'`), `cuda()`, etc), and the results of any operations 
on them are *new tensors* allocated on the GPU (a handle is returned tho so you can still print or access the tensors). Pretty much only tensor operations with 
a **trailing underscore** in the name (ex. `zero_`) *modify a tensor in place*

3. Pretty much all **tensor operations** in PyTorch (most are in the `torch` module) can execute on the CPU as well as on the GPU, with no change in the code, 
because of a **dispatching mechanism** 


## Subchapter Summaries
### 3.1 The world as floating-point numbers
* Floating-point numbers are the way networks deal with information
* PyTorch introduces a fundamental data type: the tensor (aka multidimensional array)
* PyTorch tensors are similar to NumPy multidimensional arrays, except that PyTorch tensors have the ability to be put on GPUs, 
distribute operations on multiple devices/machines, and keep track of computations

### 3.2 Tensors: multidimensional arrays
*  PyTorch tensors are much more efficient that Python lists
*  For Python lists, each element is a Python object individually randomly allocated in memory (boxed)
*  For tensors (or NumPy arrays), the entire tensor is a view over a contiguous memory block, where each element is pretty much just a 32 bit (4 byte) float 
(unboxed C numeric type → close to the machine). Accessing elements of a tensor does not allocate a new chunk of memory or create a new tensor object, it just shows a different view (subset) of the underlying tensor

### 3.3 Indexing tensors
* You can index tensors the same way as Python lists or NumPy arrays (i.e. range indexing)
* PyTorch also supports advanced indexing (see in Chapter 4)

### 3.4 Named tensors
* PyTorch 1.3 added named tensors as an experiment

### 3.5 Tensor element types
* https://pytorch.org/docs/stable/tensors.html
* Objects within a tensor must all be numbers of the same type. If you mix input types in an operation, all the inputs are converted to the larger type.
* The default tensor type is float32 (32-bit floating-point, also just float), but if you create a tensor with integers as arguments it will be type int64 (because you can use tensors as indexes in other tensors) 
* You can manage the dtype attribute in multiple ways, but using to for casting is preferred because it can take additional arguments (like device).

### 3.6 The tensor API
* Most operations for tensors are in the torch module, but most can also be called as methods to the tensor object 
* There are lot of tensor operations which are well organized in the docs: http://pytorch.org/docs.

### 3.7 Tensors: scenic views of storage
* Values in tensors are allocated to contiguous chunks of memory managed by torch.Storage and can be indexed as a 1-D array (and can modified via indexing them)
* The underlying values are allocated to memory only once → multiple tensors can index the same storage (this much more efficient than Python)
* Tensor operations usually create a new tensor and leave the source tensor unchanged, EXCEPT for tensor operations with a trailing underscore in the name (ex. zero_) which modify a tensor in place 

### 3.8 Tensor metadata: size, offset, and stride
* Along with the storage, tensors are defined by their size (shape), offset (index of first element of this tensor; is 0 if not a subtensor), and stride (number of elements to be skipped to obtain next element along each dimension) 
* Many tensor operations are inexpensive (e.g. transposing or extracting a subtensor) because you only need to change the size, offset, and stride values
* You can reshuffle a non-contiguous tensor to a contiguous one using the contiguous method (Why? because its more efficient for CPUs and some tensor operations only work on contiguous tensors, such as view)

### 3.9 Moving tensors to the GPU
* PyTorch pretty much only works on GPUs that have support for CUDA (although they are working on TPU support for the future)
* You can copy tensors from CPU RAM to GPU RAM using various methods (to(device='cuda:0'), cuda(), etc), and the results of any operations on them are new tensors allocated on the GPU (a handle is returned tho so you can still print or access the tensors)
* You can move the GPU tensors back to the CPU (to(device='cpu'), cpu())

### 3.10 NumPy interoperability
* PyTorch tensors can be converted to NumPy arrays really easily using numpy() method (no cost at all if on CPU). If the tensor is on the GPU, PyTorch copies the contents to a NumPy array on the CPU
* It is also easy to convert NumPy arrays to PyTorch tensors using from_numpy() method, but be careful because NumPy arrays are dtype double(float64) by default (we usually want float(float32) for tensors)

### 3.11 Generalized tensors are tensors, too
* PyTorch computations can be called on tensors on CPUs or GPUs (or others) because of a dispatching mechanism that hooks up the user API (the methods we call) to the correct backend functions (actually doing the calculations)
* This means all tensor operations in PyTorch can execute on the CPU as well as on the GPU, with no change in the code.

### 3.12 Serializing tensors
* we can easily serialize/save tensors to .t files, but this file format is can only be read by PyTorch
* We can save to HDF5 format instead after converting a tensor to a NumPy array https://pytorch.org/docs/stable/tensors.html
