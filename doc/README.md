# SYCL_FS Api

# 0. Presentation

See [README.md](../README.md)

# 1. Initialisation of the API

The `fs` API requires an initialisation to be done on the host. Remote file access can be achieved in various ways. One could use hardware features of the `device` such as NVIDIA's `GPUdirect Storage`
. This requires to program a filesystem driver in the remote memory access engine of the GPU. Another way is to use _asynchronous remote function calls_ through a shared memory and remote memory
accesses to move the data.

The API exposes, for performance and memory management reasons, a limited number of communication channels (`channel_count`)
which are to be managed by the user (it can be as simple as using the `nd_item`'s `global_linear_id`). For the same reasons, the API is typed (`T`), otherwise we should be copying data using `char`s
to avoid memory alignment issues. Each communication channel comes with a buffer which can contains up to `buffer_len` elements (of type `T`). This limits the maximum size of a single read or write.

In the case of an implementation using _asynchronous remote function calls_, the remote function **runner** will be calling blocking functions which will prohibit parallelism. To avoid this issue, we
can allow the remote function runner to create I/O threads with the template parameter `parallel_host_file_io` set to `true` (default). On devices using this implementation, to avoid potential
starvation caused by the **runner** thread, a `frequency` in Hz can be set. This frequency determines the maximal number of times a channel will be executed (if needed to) by the host, each second.

Some devices support performing I/O with device memory, bypassing the host's CPU. The support depends on both the device and the filesystem where our files are located. A query later presented allows
the user to know whether the pair "device-filesystem" allow this mode of operating. If supported, the template parameter `use_dma` can be set to `true`. This will allow the runtime to initialize the
DMA engine. Finally, only when using DMA, the template parameter `use_pinned_memory` is controlling whether the I/O calls will use user-allocated device-buffers or runtime-allocated ones. This
parameter is important as pinning memory to perform DMA I/O is expensive. When `use_pinned_memory` is set to `true`, the API will provide a query to get a I/O buffer related to a channel. The user
will also lose the ability to specify the device memory pointer when doing the I/O calls. Only a buffer offset will be providable. If set to `false`, memory will be pinned and unpinned for every I/O
call which can be expensive, but the API will be the same as the one without pinned memory.

The API is thread safe.

### Note

Setting this parameter to true does not mean that the runner will necessarily launch threads, we're just **allowing** it to.

### Sum-up

| `par_host_file_io` | `use_dma`\* | `use_pinned_mem` |           Data flow            |                 Pros                 |                         Cons                         |
| :----------------: | :---------: | :--------------: | :----------------------------: | :----------------------------------: | :--------------------------------------------------: |
|      `false`       |   `false`   |       `X`        | storage <-> host OS <-> device |         Works everywhere\*\*         |            The host serializes the calls             |
|      `false`       |   `true`    |     `false`      |       storage <-> device       |         Data bypasses the OS         | Host blocks until DMA IO succeeded & cost of pinning |
|      `false`       |   `true`    |      `true`      |       storage <-> device       | Good performances and less CPU usage |      Host runner blocks until DMA IO succeeded       |
|       `true`       |   `false`   |       `X`        | storage <-> host OS <-> device |   Works everywhere\*\* in parallel   |                    More CPU usage                    |
|       `true`       |   `true`    |     `false`      |       storage <-> device       |   Bypassing the OS and easy syntax   |                Cost of pinning memory                |
|       `true`       |   `true`    |      `true`      |       storage <-> device       |     Bypassing the OS, best perf      |            Forced to use provided buffers            |

- `X` means not applicable.
- \* check device+filesystem support with `fs::has_dma`.
- \* \* everywhere where `fs::has_support` is `true`.
- All calls can be emitted in parallel from a kernel.
- A reasonable optimization on a Host/CPU device would be to use "DMA" even if the user didn't specify using DMA as this avoids useless memory copies and does not change the compatibility nor the
  observable behaviour.

## 1.1 `fs` interface

```C++
template<typename T, bool parallel_host_file_io = true, bool use_dma = false, bool use_pinned_memory = false>
class fs {
public:
    /// Constructor
    fs(const sycl::queue& q, size_t channel_count, size_t buffer_len, double frequency = 100000);

    /// Accessors
    fs_accessor<T, use_dma, use_pinned_memory> get_access() const;
    fs_accessor_work_group<T, use_dma, use_pinned_memory> get_access_work_group(sycl::handler& cgh) const;

    /// Queries
    static size_t required_host_alloc_size(const sycl::queue& q, size_t channel_count, size_t buffer_len);
    static size_t required_device_alloc_size(const sycl::queue& q, size_t channel_count, size_t buffer_len);
    static size_t required_local_alloc_size_work_group(const sycl::queue& q, size_t channel_count, size_t buffer_len);
    static bool has_support(const sycl::queue& q);
    static bool has_dma(const sycl::queue& q, const std::string& file = {});
};
```

## 1.2 Methods description

| Constructor                                                                                           | Description                                                                                                                                                                                                                                      |
| ----------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `fs(const sycl::queue& q, size_t channel_count,`<br/>`size_t buffer_len, double frequency = 100000);` | `q` is the queue that is going to be used to access the filesystem on which we will initialise the API. <br/> `channel_count` is the number of channels to create, each will be able to do a single read/write of `buffer_len` elements maximum. |

| Methods                                                                | Description                                                                                                                                                |
| ---------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `fs_accessor<T> get_access() const;`                                   | Returns an `fs_accessor<T>` which can be used in a kernel to open files.                                                                                   |
| `fs_accessor_work_group<T> get_access_work_group(sycl::handler& cgh);` | Returns an `fs_accessor_work_group<T>` which has to be acquired in a command group scope which can the be used in a kernel to open a file on a work group. |

### Queries

They are to be called with the same template parameters and arguments as one would want to call the constructor: `const sycl::queue& q, size_t channel_count, size_t buffer_len`

| Static methods                                                      | Description                                                                                                                                      |
| ------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| `required_host_alloc_size(...);`                                    | Returns the allocation size which will have to be performed on the host by the implementation. Parameters are the same as for the constructor.   |
| `required_device_alloc_size(...);`                                  | Returns the allocation size which will have to be performed on the device by the implementation. Parameters are the same as for the constructor. |
| `required_local_alloc_size_work_group(...);`                        | Returns the size of the local memory that will be used for each work group when opening files with `get_access_work_group`.                      |
| `bool has_support(const sycl::queue& q);`                           | Returns whether the queue supports this API                                                                                                      |
| `bool has_dma(const sycl::queue& q, const std::string& file = {});` | Returns whether the queue and the file system of the file the supports DMA together.                                                             |

## Improvements

Adding a batch constructor that takes a list/vector/array of `std::string` to allow iterating over files in the kernel? Or maybe _format strings_ and a range that can then be used to produce file
names.

# 2. The filesystem accessor.

Once the accessor acquired, it can be used in the kernel to open a file. Two accessor types do exist: `fs_accessor<T>` and `fs_accessor_work_group<T>`. The filenames are null-terminated strings. All
files are opened in binary mode.

The user has to provide a channel index when opening a file. The channel index can be reused, but one channel cannot be used in parallel to open a file. Using a single channel across all the kernel
will thus serialise all the file I/O.

An accessor can be used to open any number of files using the channels you want. They can all be used in parallel. The return values are the number of elements saved or read.

## 2.1 File opening modes:

The `fs_mode` is mapped to the posix opening modes, but it's wrapped in this `enum class` to abstract the platform. It's defined as:

```c++
enum class fs_mode {
   read_only /** R */,
   write_only /** W */,
   append_only /** A */,
   read_write /** R+ */,
   erase_read_write /** W+ */,
   append_read /** A+ */,
   none,
};
```

## 2.2 Single work-item accessor `fs_accessor<T, ...>`

This accessor offers the following interface:

```C++
template<typename T, bool use_dma = false, bool use_pinned_memory = false>
class fs_accessor{
    template<fs_mode mode>
    std::optional<fs_descriptor<T, mode, use_dma, use_pinned_memory>> open(size_t channel_idx, const char* filename) const;
    std::optional<sycl::range<2>> load_image(size_t channel_idx, const char *filename, sycl::accessor<sycl::uchar4, 2, mode, target> image_accessor) const;
};
```

This accessor is the single-threaded one, but is the easiest to use. It can only be called from single a work-item at a time. On opening success, the `std::optional` returned contains a file
descriptor which can be used to perform I/O on the file and to close it.

If this accessor is used in a `nd_range` kernel, one should be really careful to only use it with a single work-item from the work-group. Trying to open a file with several work-items will have the
same (potentially undefined) behaviour as opening a file several times with `fopen`. Writing to the file will be wrong. One should also be careful to use a different `channel_idx` for each thread that
opens a file, even if it's several times the same one.

`load_image` allows to load a picture into a `sycl::accessor<sycl::uchar4, 2, ...>` that has a compatible access mode. The continuous dimensions corresponds to a line. Each pixel is encoded in a `sycl::uchar4` (vector of four unsigned chars) in the order **RGBA**.
We do not use `sycl::image` as DPC++ doesn't seem to support `read_write` accessors. Having a read/write only accessor won't really allow to use that library in a useful manner.
The user must ensure that the buffer size, ie `buffer_len * sizeof(T)` can store the whole picture that has a size equal to `sizeof(sycl::uchar4) * x * y`. If that's not true, the 
call will fail as a check is performed on the host to prevent overflows. That issue comes from the fact that we don't know the picture size before decoding it. 
RMA mode is not handled as we cannot use the 2D `sycl::accessor` safely from the CPU.
The call returns `std::optional<sycl::range<2>>` that contains the picture size, on success in the SYCL order, ie: `(y_size, x_size)`.

## 2.3 Parallel accessor `fs_accessor_work_group<T, ...>`

```C++
template<typename T, bool use_dma = false, bool use_pinned_memory = false>
class fs_accessor_work_group : protected fs_accessor<T, use_dma, use_pinned_memory> { // Up-cast possible
    template<fs_mode mode>
    std::optional<fs_descriptor_work_group<T, mode, use_dma, use_pinned_memory>> open(sycl::nd_item<1> item, size_t channel_idx, const char* filename) const;
};
```

This accessor is to be used in a `nd_range` kernel and be called by ALL the work-items, from the work group. Each work item will call this method, but only the arguments passed by the first work item
will be used to open the file. Every work item will get a file descriptor pointing to the same file.

When opening a file, each work-item passes it's `nd_item`, to `open` method. The resulting file descriptors will be later used to perform faster I/O using all the work-items from a work-group. This
will allow taking full advantage of GPU hardware as no thread will be left unused.

This accessor and `fs_descriptor_work_group` will use work-group local memory to sync data between the work-items.

When using DMA, this accessor has no advantage over the base one.

## 2.4 Common queries

| Methods                       | Description                                                        |
| ----------------------------- | ------------------------------------------------------------------ |
| `size_t get_channel_count();` | Returns the number of channel available to perform I/O operations. |
| `size_t abort_host();` | Will initiate termination of the program. First all pending operations will be completed, threads joined before calling `abort()`. There might be some delay |

# 3. Performing I/O

## 3.1 Specifying the offset

The available values to the parameter `offset_type` are:

```C++
enum class fs_offset {
        begin /** Does the offset from the start of the file */,
        end /** From the end */,
        current /** From the current position */,
};
```

The value `offset` then allows to specify how many elements we want to move forward/backwards from the whence.

## 3.2 I/O using the single work-item file descriptor (`fs_descriptor`) and `use_pinned_memory=false` :

The file descriptor which is returned if we opened the file using `open(...)` has the following interface:

```C++
template<typename T, fs_mode open_mode, bool use_dma, bool use_pinned_memory>
class fs_descriptor {
public:
    // Write methods
    size_t write(const T* device_src, size_t elt_count, fs_offset offset_type = fs_offset::current, int32_t file_offset = 0);

    template<...>
    size_t write(const sycl::accessor<T, 1, ...>& accessor, size_t accessor_begin, size_t count, fs_offset offset_type = fs_offset::current, int32_t file_offset = 0);

    //Read methods
    size_t read(T* device_dst, size_t elt_count, fs_offset offset_type = fs_offset::current, int32_t file_offset = 0);

    template<...>
    size_t read(sycl::accessor<T, 1, ...>& dst_accessor, size_t accessor_begin, size_t elt_count, fs_offset offset_type = fs_offset::current, int32_t file_offset = 0)

    // Close
    void close();
};
```

All the methods are to be called by a single work-item at the same time. We cannot write and read in parallel using this accessor.

| Methods                                                                                                                                      | Description                                                                                                                                                                     |
| -------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `write(const T* device_src, size_t elt_count, fs_offset offset_type, int32_t file_offset);`                                                  | Writes `elt_count` elements of type `T` from `device_src` to the file.                                                                                                          |
| `write(const sycl::accessor<T, 1, ...>& src_accessor, size_t accessor_begin, size_t elt_count, fs_offset offset_type, int32_t file_offset);` | Writes `elt_count` elements of type `T` from the `sycl::accessor`, with an accessor offset of `accessor_begin` to the file. The accessor must be 1-dimensional and of type `T`. |
| `read(T* device_dst, size_t elt_count, fs_offset offset_type, int32_t file_offset);`                                                         | Reads `elt_count` elements of type `T` to `device_dst` from the file.                                                                                                           |
| `read(sycl::accessor<T, 1, ...>& dst_accessor, size_t accessor_begin, size_t elt_count, fs_offset offset_type, int32_t file_offset);`        | Reads `elt_count` elements of type `T` to the `sycl::accessor`, with an accessor offset of `accessor_begin` from the file.The accessor must be 1-dimensional and of type `T`.   |
| `close();`                                                                                                                                   | Closes the file. The file descriptor cannot be reused.                                                                                                                          |

## 3.3 I/O using the parallel file descriptor (`fs_descriptor_work_group`) and `use_pinned_memory=false`:

**These methods are to be called by the whole work-group we used to open the file.**
They allow to use all the work-items to speedup I/O, they perform the needed synchronisations between the work-items and the I/O is done using packed memory accesses.

```C++
template<typename T, fs_mode open_mode, bool use_dma, bool use_pinned_memory>
class fs_descriptor_work_group {
public:
    // Write methods
    size_t write(const T* device_src, size_t elt_count, fs_offset offset_type = fs_offset::current, int32_t offset = 0);

    template<...>
    size_t write(const sycl::accessor<T, 1, ...>& src_accessor, size_t accessor_begin, size_t count, fs_offset offset_type = fs_offset::current, int32_t offset = 0);

    //Read methods
    size_t read(T* device_dst, size_t elt_count, fs_offset offset_type = fs_offset::current, int32_t offset = 0);

    template<...>
    size_t read(sycl::accessor<T, 1, ...>& dst_accessor, size_t accessor_begin, size_t count, fs_offset offset_type = fs_offset::current, int32_t offset = 0);

    // Close
    void close();
};
```

**Only the arguments passed by the first work-item are considered.**

| Methods                                                                                                                                      | Description                                                                                                                                                                     |
| -------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `write(const T* device_src, size_t elt_count, fs_offset offset_type, int32_t file_offset);`                                                  | Writes `elt_count` elements of type `T` from `device_src` to the file.                                                                                                          |
| `write(const sycl::accessor<T, 1, ...>& src_accessor, size_t accessor_begin, size_t elt_count, fs_offset offset_type, int32_t file_offset);` | Writes `elt_count` elements of type `T` from the `sycl::accessor`, with an accessor offset of `accessor_begin` to the file. The accessor must be 1-dimensional and of type `T`. |
| `read(T* device_dst, size_t elt_count, fs_offset offset_type, int32_t file_offset);`                                                         | Reads `elt_count` elements of type `T` to `device_dst` from the file.                                                                                                           |
| `read(sycl::accessor<T, 1, ...>& dst_accessor, size_t accessor_begin, size_t elt_count, fs_offset offset_type, int32_t file_offset);`        | Reads `elt_count` elements of type `T` to the `sycl::accessor`, with an accessor offset of `accessor_begin` from the file.The accessor must be 1-dimensional and of type `T`.   |
| `close();`                                                                                                                                   | Closes the file. The file descriptor cannot be reused.                                                                                                                          |

## 3.4 I/O using the single threaded file descriptor and `use_pinned_memory=true`:

```C++
template<typename T, fs_mode open_mode, bool use_dma, bool use_pinned_memory>
class fs_descriptor {
public:
    // Get pinned buffer ptr
    T* get_device_io_buffer();

    // Write methods
    size_t write(size_t device_buff_offset, size_t elt_count, fs_offset file_offset_type = fs_offset::current, int32_t file_offset = 0);

    //Read methods
    size_t read(size_t device_buff_offset, size_t elt_count, fs_offset file_offset_type = fs_offset::current, int32_t file_offset = 0);

    // Close
    void close();
};
```

| Methods                                                                                                       | Description                                                                                                                        |
| ------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| `T* get_device_io_buffer();`                                                                                  | Gets a pointer to the device's pinned IO buffer where one could write or store data before calling the `write` or `read` function. |
| `size_t write(size_t device_buff_offset, size_t elt_count, fs_offset file_offset_type, int32_t file_offset);` | Writes `elt_count` elements of type `T` from the device's buffer, with device buffer offset of `device_buff_offset` to the file.   |
| `size_t read(size_t device_buff_offset, size_t elt_count, fs_offset file_offset_type, int32_t file_offset);`  | Reads `elt_count` elements of type `T` to the device's buffer, with device buffer offset of `device_buff_offset` from the file.    |
| `close();`                                                                                                    | Closes the file. The file descriptor cannot be reused.                                                                             |

### Note

When using DMA with or without pinned memory, the parallel file descriptors have no performance advantage over the serial ones. They still exist for ease of use.

## 3.5 Common queries

| Methods                             | Description                                                                                          |
| ----------------------------------- | ---------------------------------------------------------------------------------------------------- |
| `size_t get_max_single_io_count();` | Returns the maximum number of elements of type `T` that can be read or written in a single I/O call. |

## 3.6 Improvements

Add `_async` write and read methods to asynchronously perform I/O. Right not this is not implementable with the RPC backend, but with DMA it should be.

# 4. Use examples in a `nd_range` kernel

For working examples see [the benchmark](../examples/fs_benchmark.cpp).

```C++
sycl::fs<float> fs(q, work_groups, file_size); // One channel per work group, max single read of file_size
q.submit([&](sycl::handler& cgh) {
    /* To create the parallel file accessor, we need to pass the sycl::handler in order to get access to local memory (shared within a work group) */
    sycl::fs_accessor_work_group<float> parallel_accessor = fs.get_access_work_group(cgh);

    /* Launching the kernel */
    cgh.parallel_for<processing_kernel>(sycl::nd_range<1>(work_items * work_groups, work_items), [=](sycl::nd_item<1> item) {
        const size_t work_group_id = item.get_group_linear_id();
        const size_t work_item_id = item.get_local_linear_id();
        const size_t channel_idx = work_group_id; // We use work_group_id as the channel index.
        float* wg_buffer = ... ; // Get some buffer for the work group

        /* Iterating over the pictures that are to be processed by the current work group */
        const char* filename_ptr = ... ; // something
        auto fh = parallel_accessor.open<sycl::fs_mode::read_only>(item, channel_idx, filename_ptr);
        if (fh) {
            fh->read(wg_buffer, file_elt_count); // Read the file using all the work items
            fh->close();
        }

        /* Process your floats using all the work items in parallel*/

        fh = parallel_accessor.open<sycl::fs_mode::write_only>(item, channel_idx, filename_ptr);
        if (fh) {
            fh->write(wg_buffer, file_elt_count); // Write the file using all the work items
            fh->close();
        }
    });
}).wait();
```

# SYCL_RPC Api

### Important notes

- One could allow running the host functions in separate threads by setting `parallel_runners` to `true`. To disallow some functions from spawning a thread, `.set_allowed_to_spawn_host_thread(false)`
  can be called on a communication channel.
- A frequency (positive double) can be set to limit the speed with which the host "main runner" scans the channels for RPC to execute. The frequency is the number of complete scans per second. If the
  frequency is negative, the host runner will run in a loop until it gets scheduled.
- If we want, a function can be called synchronously on the device. but the call will be blocking, especially if the frequency is slow, and the host called function is itself blocking. This is
  independent from `set_allowed_to_spawn_host_thread`
- A channel can be pre-acquired and released after the async call. This allows the user to communicate data outside the function arguments. We could write to a host buffer which is identified by the
  channel index. That is exactly how the file system API was built.
- A synchronous call always returns an `std::optional` which, on success, contains the value. The communication channel is always automatically closed unless asked otherwise.
- An asynchronous call won't close the channel as we need to query it later to get the result. Using the value returned, on success, in the `std::optional` from the rpc call is UB. One need to use the
  async functions: `try_get_result`, `get_result` and `wait`. After getting the result one need to `release` the channel.
- Template parameters of `call_remote_procedure`: `do_acquire_channel` and `do_release_channel` specify whether the call manages the channel lifetime. By default, both are set to `true`.
- Even though we're talking about "remote procedure calls", they are in fact "remote function calls", the communication is bidirectional.
- Releasing twice a channel or releasing a not acquired channel is a programming error and will `abort()` by safety.
