# Filesystem API for SYCL and Parallel Asynchronous Remote Procedure Calls

This little project provides an implementation of a storage API for SYCL. The particularity of this API is that the calls can be emitted straight from a running SYCL kernel, without any interruption.
This API communicates with the host using remote procedure calls. It is designed to leverage hardware capabilities of devices such as
[NVIDIA's GPUDirect Storage](https://developer.nvidia.com/blog/gpudirect-storage/) which completely bypasses the host and allows the GPU to store and read data from a compatible file system. Without
the DMA support, this api works in an emulated mode and uses the host's OS to store data. This emulated mode works on any filesystem, unlike the DMA one.

## Features

- Concurrent filesystem I/O from all the work-groups and work-items of a SYCL parallel kernel, at the same time.
- Can leverage device's hardware IO features to bypass the host OS.
- Possibility to open a file across a whole work-group to increase throughput and avoid the **divergence of the control flow**.
- USM and accessor interface.
- Dynamic threaded access to the underlying filesystem (enabled by default).
- I/O latency control to reduce pressure on the CPU.
- POSIX functionalities (opening modes and seek)
- Type-safe & Header-based API.

## Target applications

- Machine Learning: Using this API, one can train a network without having to ever stop the kernel to load new datasets. The SYCL kernel will be able to load the datasets, let's say pictures, itself.
  One will also be able to save the trained network at runtime (which would even allow _inter-kernel_ communication).
- Real-time processing: one could open some character devices in parallel and process the data in real time. Could be image recognition on a video feed.
- Batch processing: See the example. With a single kernel launch and one fixed-size memory allocations, we can process an unbound number of files on the device (without having to re-alloc or re-launch
  kernels).
- Processing files that do not fit in the memory of the GPU: to process a petabyte dataset, on a regular GPU one would have to launch tons of kernels and manage the data. What if the kernel has to
  perform random accesses on the file? Now it can all be done from the kernel.
- And many more!

### Async RPC features

- Parallel API: all work-items can call functions at the same time.
- From a SYCL kernel, one can choose to perform a function call in synchronous or asynchronous manner.
- The host can answer to the function calls in a synchronous or asynchronous manner.
- When calling a host function from a SYCL kernel, one can choose whether the host will spawn a thread to answer the call, if the functions expensive or blocking.
- Easy to set up: the user defines the functions that will be remotely executed and provides a _runner_ function that will do the call on the host (probably a big `switch`).
- Ability to choose the frequency with which the HOST will be answering the function calls, to avoid starvation.

## Performance

The benchmark (in this repository) is able to get to a bandwidth of 17 GiB/s when reading files with an NVIDIA GPU (thanks to filesystem caching on the host). The same values can be observed on the
CPU with OpenCL which suggests that the implementation is bound by the host's storage-controller/hdd

## Read the doc

The detailed API documentations is [here](doc/README.md).

## Example (see demo_fs.cpp)

As with buffers, we create a `fs` object on the host and then pass an accessor to the SYCL kernel:

```c++
#include "sycl_fs.hpp"
...
auto q = sycl::queue(sycl::gpu_selector{});
sycl::fs<T> reader(q, parallel_channel_count, max_read_len);

q.submit([&](sycl::handler &cgh) {
    auto acc = reader.get_access();
    cgh.single_task([=]() {
        ...
    });
});
```

Right now the implementation supports only a fixed number of parallel communication channels. A channel cannot be accessed twice simultaneously. But one could use the work group ID to ensure there's
no conflict (and to load a file into local memory, with a single thread, and run the computation on the work group).

The host frequency (see later) is set to 100000Hz (probably too much) which means that we cannot perform more than 100000 functions calls per channel, per second. This is done to avoid starvation.

### Opening a file

In a kernel, on the device, a file can be opened with:

```c++
auto fh = acc.open<sycl::fs_mode::read_write>(0, "my_file.txt");
```

It returns a `std::optional<fs_descriptor...>` containing the file descriptor on success

### Doing IO

```c++
if(fh){ // Checking whether the file was successfully opened
    size_t number_written = fh->write(a_message, message_size);
    size_t number_read = fh->read(a_buffer, message_size, 0); //specifying the offset
    fh->close();
}
```

All the sizes corresponds to the number of elements of type `T` that we want to process/were processed.

The `fs_mode` is mapped to the posix ones, but it's wrapped in this `enum class` to abstract the platform.

When creating the `fs` object, one could set the template parameter `parallel_host_file_io` to `true`. This will result in the parallel execution of the Remote Procedure Calls on the host.

## Remarks

Tested and working on hipSYCL with CUDA and OpenMP backends as well as on DPC++ with OpenCL and CUDA backends.

With CUDA something interesting is happening with the synchronous version of the RPC demo. With hipSYCL, every thread is executed in parallel while with DPC++, a barrier seems to be added between the
synchronous function calls and the prints. See comment in code.

It has been tested with up to 30 000 simultaneous RPC calls to CPU blocking functions (calls in a parallel_for), done from a `sycl::device` using the CUDA backend, with the parallel runners. With more
parallel calls, the OS starts complaining about too many threads.

## Building

Change the device selector if needed. Then in a build folder:

### hipSYCL

`hipSYCL_dir=/path/to_install cmake .. -DHIPSYCL_TARGETS="omp;cuda:sm_75"`

### DPC++

`CXX=dpcpp_compiler cmake ..`
Edit the cmake to change the targets.

## Work in progress

- Socket support
- Asynchronous file descriptors in the kernel: useful only in DMA mode, else there should be no advantage.
