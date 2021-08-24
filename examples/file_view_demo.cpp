#define IMAGE_LOAD_SUPPORT

#include <sycl_fs.hpp>
#include <tools/sycl_queue_helpers.hpp>

class load_image_kernel;

void launch_image_generator(sycl::queue &q, size_t max_x = 4000, size_t max_y = 4000) {
    size_t storage_required = max_x * max_y * sizeof(sycl::char4);
    //auto src = new sycl::char4[storage_required];
    //sycl::image<2> srcImage(src, sycl::image_channel_order::rgba, sycl::image_channel_type::unsigned_int8, sycl::range<2>(max_x, max_y));

    /* We initialise the file system api on the host */
    sycl::fs<uint8_t, true> fs(q, 1, storage_required);

    q.submit([&](sycl::handler &cgh) {
        sycl::fs_accessor<uint8_t> storage_accessor = fs.get_access();
        //auto acc = srcImage.get_access<sycl::uint4 , sycl::access::mode::read>(cgh);
        sycl::stream os(1024, 256, cgh);
        cgh.single_task<load_image_kernel>([=]() {
            auto result = storage_accessor.load_image(0, "my_file000002.bmp");
            if (result) {
                os << result->get(0) << " " << result->get(1) << sycl::endl;
            }
        }); // single_task
    }).wait();
}


int main(int, char **) {
    sycl::queue q = try_get_queue(sycl::gpu_selector{});
    std::cout << "Running on: " << q.get_device().get_info<sycl::info::device::name>() << std::endl;
    launch_image_generator(q);
}

