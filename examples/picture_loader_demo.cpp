#define IMAGE_LOAD_SUPPORT

#include <sycl_fs.hpp>
#include <tools/sycl_queue_helpers.hpp>

class load_image_kernel;

template<typename T = char>
void launch_image_generator(sycl::queue &q, size_t max_x = 4000, size_t max_y = 4000) {
    size_t tmp_space = max_x * max_y * sizeof(sycl::uchar4) / sizeof(T);
    sycl::buffer<sycl::uchar4, 2> image_buffer(sycl::range<2>(max_x, max_y));

    /**
     * We initialise the file system api on the host
     * The type of that API is independent of the one used to read the picture
     * tmp space must be enough to hold the picture in bytes, but it's not related
     */
    sycl::fs<T, true> fs(q, 1, tmp_space);

    q.submit([&](sycl::handler &cgh) {
        sycl::fs_accessor<T> storage_accessor = fs.get_access();
        auto buffer_accessor = image_buffer.get_access<sycl::access::mode::read_write, sycl::access::target::global_buffer>(cgh);
        sycl::stream os(1024, 256, cgh);
        cgh.single_task<load_image_kernel>([=]() {
            /* Loading the image into our 2D accessor */
            if (auto result = storage_accessor.load_image(0, "my_file000002.bmp", buffer_accessor)) {
                os << "Successfully loaded with dimensions: " << result->get(0) << "x" << result->get(1) << sycl::endl;
            } else {
                os << "Failed opening the picture" << sycl::endl;
            }
        }); // single_task
    }).wait();
}


int main(int, char **) {
    sycl::queue q = try_get_queue(sycl::gpu_selector{});
    std::cout << "Running on: " << q.get_device().get_info<sycl::info::device::name>() << std::endl;
    launch_image_generator(q);
}

