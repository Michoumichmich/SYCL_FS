/**
 * Example using the parallel work group writers
 * this leads to a huge performance improvement on GPUs over `bmp_processing.cpp`
 */


#include <sycl/sycl.hpp>
#include <sycl_fs.hpp>
#include <tools/sycl_queue_helpers.hpp>
#include <tools/usm_smart_ptr.hpp>
#include <tools/bmp_io.hpp>
#include <tools/scope_chrono.hpp>

class generating_kernel;

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

size_t get_alloc_count(size_t width, size_t height)
{
    return width * height + 4 - ((width * height) % 4);
}

/**
 * Each work group will process several pictures, one at a time, and then only all work-items will write the picture to the hard drive.
 * This means that we can keep the number of parallel writers small this reduce the number of channels needed, reduce memory, and more importantly:
 * don't have to stop the gpu from running. One could notice that the memory allocation size is independent from the number of pictures to process, as
 * well as the number of kernel submissions.
 */
size_t launch_image_generator(size_t file_count, sycl::queue& q, size_t work_groups, size_t work_items, const usm_shared_ptr<char, alloc::shared>& filenames, size_t filename_size, size_t width,
        size_t height)
{

    /* We initialise the file system api on the host  */
    sycl::fs<uint8_t, true> fs(q, work_groups, bmp::get_buffer_size(width, height));

    /* Allocating buffer for processing */
    usm_shared_ptr<pixel, alloc::device> gpu_image_buffer(get_alloc_count(width, height) * work_groups, q);

    q.submit([&, filenames = filenames.raw(), gpu_image_buffer = gpu_image_buffer.raw()](sycl::handler& cgh) {

        /* To create the parallel file accessor, we need to pass the sycl::handler in order to get access to local memory (shared within a work group) */
        sycl::fs_accessor_work_group<uint8_t> image_writer = fs.get_access_work_group(cgh);
        sycl::stream os(1024, 256, cgh);
        cgh.parallel_for<generating_kernel>(sycl::nd_range<1>(work_items * work_groups, work_items), [=](sycl::nd_item<1> item) {
            const size_t work_group_id = item.get_group_linear_id();
            const size_t work_item_id = item.get_local_linear_id();
            const size_t channel_idx = work_group_id;
            pixel* work_group_image_buffer = gpu_image_buffer + channel_idx * get_alloc_count(width, height); // Each wg has its own memory region

            /* Iterating over the pictures that are to be processed by the current work group */
            for (size_t processed_file_id = work_group_id; processed_file_id < file_count; processed_file_id += work_groups) {
                const char* filename_ptr = filenames + filename_size * processed_file_id; // Getting the file name pointer

                /* Writing dummy data to the buffer with the work items, in a packed manner */
                for (size_t i = work_item_id; i < width * height; i += work_items) {
                    work_group_image_buffer[i] = yuv_2_rgb((50 * work_group_id + i % width) % 256, (50 * work_group_id + i / height) % 256, 150);
                }

                /* Writing the picture with all the work items in parallel. The first work item will open/close the file, but
                 * the actual writing will be done in parallel, using all the work items */
                if (!bmp::save_picture_work_group(item, channel_idx, image_writer, filename_ptr, width, height, work_group_image_buffer)) {
                    os << "Failure saving: " << filename_ptr << sycl::endl;
                }
            } // Back to for loop over the pictures
        }); // parallel_for
    }).wait();
    std::cout << "Generating pass done!" << std::endl;
    return 3 * width * height * file_count; // Data processed
}

/**
 * Little example of how to read files (fundamentally the same as the previous one)
 */
size_t launch_image_checker(size_t file_count, sycl::queue& q, size_t work_groups, size_t work_items, const usm_shared_ptr<char, alloc::shared>& filenames, size_t filename_size, size_t width,
        size_t height)
{
    sycl::fs<uint8_t> fs(q, work_groups, bmp::get_buffer_size(width, height));

    /* Allocating buffer for processing */
    usm_shared_ptr<pixel, alloc::device> device_image_buffer(get_alloc_count(width, height) * work_groups, q);

    q.submit([&, filenames = filenames.raw(), device_image_buffer = device_image_buffer.raw()](sycl::handler& cgh) {
        /* To create the parallel file accessor, we need to pass the sycl::handler in order to get access to local memory (shared within a work group) */
        sycl::fs_accessor_work_group<uint8_t> image_accessor = fs.get_access_work_group(cgh);
        sycl::stream os(1024, 256, cgh);
        cgh.parallel_for<processing_kernel>(sycl::nd_range<1>(work_items * work_groups, work_items), [=](sycl::nd_item<1> item) {
            const size_t work_group_id = item.get_group_linear_id();
            const size_t work_item_id = item.get_local_linear_id();
            const size_t channel_idx = work_group_id;
            pixel* work_group_image_buffer = device_image_buffer + channel_idx * get_alloc_count(width, height);

            /* Iterating over the pictures that are to be processed by the current work group */
            for (size_t processed_file_id = work_group_id; processed_file_id < file_count; processed_file_id += work_groups) {
                const char* filename_ptr = filenames + filename_size * processed_file_id;

                /* Parallel picture loading using the work group */
                bmp::load_picture_work_group(item, channel_idx, image_accessor, filename_ptr, width, height, work_group_image_buffer);
                for (size_t i = work_item_id; i < width * height; i += work_items) {
                    const pixel expected = yuv_2_rgb((50 * work_group_id + i % width) % 256, (50 * work_group_id + i / height) % 256, 150);
                    if (work_group_image_buffer[i] != expected) {
                        os << "Error, got: " << work_group_image_buffer[i] << " instead of: " << expected;
                        abort_kernel(os);
                    }
                }
                /* Parallel picture saving using the work group */
                bmp::save_picture_work_group(item, channel_idx, image_accessor, filename_ptr, width, height, work_group_image_buffer);
            }
        });
    }).wait();
    std::cout << "Read, check and rewrite pass done!" << std::endl;
    return 2 * 3 * width * height * file_count; // Data processed
}

int main(int, char**)
{
    sycl::queue q = try_get_queue(sycl::gpu_selector{});
    std::cout << "Running on: " << q.get_device().get_info<sycl::info::device::name>() << std::endl;

    /* Settings */
    size_t files_to_process = 50;
    const char format_string[] = "my_file%06d.bmp";
    size_t width = 1920;
    size_t height = 1080;

    /* Generating the file names */
    const size_t filename_size = (size_t) std::snprintf(nullptr, 0, format_string, (int) files_to_process) + 1;
    assert(filename_size > 0 && "Wrong format string");
    usm_shared_ptr<char, alloc::shared> filenames(filename_size * files_to_process, q);
    for (size_t i = 0; i < files_to_process; ++i) {
        std::snprintf(filenames.raw() + i * filename_size, filename_size, format_string, (int) i);
    }

    /* nd_range settings are independent of the number of files to process */
    size_t work_item_count = 64; // work items
    size_t work_group_count = 24 * 4; // work groups

    /* Benchmarking and testing the batch image processor. */
    {
        scope_chrono c("Processing");
        /* Generating the pictures */
        size_t io_bytes = launch_image_generator(files_to_process, q, work_group_count, work_item_count, filenames, filename_size, width, height);
        /* Reading them, checking whether everything is correct and writing them back */
        io_bytes += launch_image_checker(files_to_process, q, work_group_count, work_item_count, filenames, filename_size, width, height);
        io_bytes += launch_image_checker(files_to_process, q, work_group_count, work_item_count, filenames, filename_size, width, height);
        std::cout << "Processed " << (double) io_bytes / (1024. * 1024.) << " MiB, bandwidth: " << (double) io_bytes / (1024. * 1024.) / c.stop() << " MiB/s " << std::endl;
    }

}

