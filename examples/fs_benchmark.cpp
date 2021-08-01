/**
 * Example using the parallel work group writers
 * this leads to a huge performance improvement on GPUs over `bmp_processing.cpp`
 */

#include <sycl/sycl.hpp>
#include <sycl_fs.hpp>
#include <tools/sycl_queue_helpers.hpp>
#include <tools/usm_smart_ptr.hpp>
#include <tools/scope_chrono.hpp>

class processing_kernel;

using namespace usm_smart_ptr;

/**
 * Forces the compiler to dereference the null pointer
 * as writing to OS is part of the observable behaviour.
 */
void abort_kernel(sycl::stream os)
{
    size_t p = 0;
    os << *reinterpret_cast<int*>((int*) p) << sycl::endl;
}

template<typename T>
size_t run_one_pass(size_t file_count, size_t file_size, sycl::queue& q, size_t work_groups, size_t work_items, const usm_shared_ptr<char, alloc::shared>& filenames, size_t filename_size)
{
    size_t file_elt_count = file_size / sizeof(T) + (file_size % sizeof(T) != 0);

    /* Allocating buffer for processing */
    usm_shared_ptr<T, alloc::device> device_buffer(file_elt_count * work_groups, q);
    sycl::fs<T> fs(q, work_groups, file_elt_count + 1);

    q.submit([&, filenames = filenames.raw(), device_buffer = device_buffer.raw()](sycl::handler& cgh) {
        /* To create the parallel file accessor, we need to pass the sycl::handler in order to get access to local memory (shared within a work group) */
        sycl::fs_accessor_work_group<T> parallel_accessor = fs.get_access_work_group(cgh);
        sycl::stream os(1024, 256, cgh);
        cgh.parallel_for<processing_kernel>(sycl::nd_range<1>(work_items * work_groups, work_items), [=](sycl::nd_item<1> item) {
            const size_t work_group_id = item.get_group_linear_id();
            const size_t work_item_id = item.get_local_linear_id();
            const size_t channel_idx = work_group_id;
            T* wg_buffer = device_buffer + channel_idx * file_elt_count;

            /* Iterating over the pictures that are to be processed by the current work group */
            for (size_t processed_file_id = work_group_id; processed_file_id < file_count; processed_file_id += work_groups) {
                const char* filename_ptr = filenames + filename_size * processed_file_id;
                auto fh = parallel_accessor.template open<sycl::fs_mode::erase_read_write>(item, channel_idx, filename_ptr);

                if (!fh) { abort_kernel(os); }
                if (fh->write(wg_buffer, file_elt_count) != file_elt_count) { abort_kernel(os); }
                if (fh->read(wg_buffer, file_elt_count, sycl::fs_offset::begin, 0) != file_elt_count) { abort_kernel(os); }
                fh->close();

            }
        });
    }).wait();
    return 2U * file_size * file_count; // Data processed
}

int main(int, char**)
{
    sycl::queue q = try_get_queue(sycl::gpu_selector{});
    std::cout << "Running on: " << q.get_device().get_info<sycl::info::device::name>() << std::endl;

    assert(sycl::fs<bool>::has_support(q) && "This queue does not support the FS api");

    /* Settings */
    size_t files_to_process = 24 * 5;
    const char format_string[] = "my_file%06d.bmp";
    size_t file_byte_size = 1024 * 1024 * 2;

    /* Generating the file names */
    const size_t filename_size = (size_t) std::snprintf(nullptr, 0, format_string, (int) files_to_process) + 1;
    assert(filename_size > 0 && "Wrong format string");
    usm_shared_ptr<char, alloc::shared> filenames(filename_size * files_to_process, q);
    for (size_t i = 0; i < files_to_process; ++i) {
        std::snprintf(filenames.raw() + i * filename_size, filename_size, format_string, (int) i);
    }

    /* nd_range settings are independent of the number of files to process */
    size_t work_item_count = 32 * 2; // work items per work group
    size_t work_group_count = 24 * 1; // work groups

    /* Benchmarking */
    {
        scope_chrono c("Processing");
        size_t io_bytes = 0;
        for (int i = 0; i < 100; ++i) {
            io_bytes += run_one_pass<int64_t>(files_to_process, file_byte_size, q, work_group_count, work_item_count, filenames, filename_size);
            std::cout << "Processed " << (double) io_bytes / (1024. * 1024. * 1024.) << " GiB, AVG bandwidth: " << (double) io_bytes / (1024. * 1024. * 128.) / c.stop() << " Gib/s." << std::endl;
        }

    }

}

