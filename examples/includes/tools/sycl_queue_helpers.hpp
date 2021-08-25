#pragma once

#include <sycl/sycl.hpp>
#include <iostream>

#ifdef USING_COMPUTECPP
class queue_kernel_tester;
namespace cl::sycl::usm{
    using cl::sycl::experimental::usm::alloc;
}
#endif

/**
 * Selects a CUDA device (but returns sometimes an invalid one)
 */
class cuda_selector : public sycl::device_selector {
public:
    int operator()(const sycl::device &device) const override {
        //return device.get_platform().get_backend() == sycl::backend::cuda && device.get_info<sycl::info::device::is_available>() ? 1 : -1;
        return device.is_gpu() && (device.get_info<sycl::info::device::driver_version>().find("CUDA") != std::string::npos) ? 1 : -1;
    }
};

/**
 * Tries to get a queue from a selector else returns the host device
 * @tparam strict if true will check whether the queue can run a trivial task which implied
 * that the translation unit needs to be compiler with support for the device you're selecting.
 */
template<bool strict = true, typename T>
inline sycl::queue try_get_queue(const T &selector) {
    auto exception_handler = [](const sycl::exception_list &exceptions) {
        for (std::exception_ptr const &e: exceptions) {
            try {
                std::rethrow_exception(e);
            }
            catch (sycl::exception const &e) {
                std::cout << "Caught asynchronous SYCL exception: " << e.what() << std::endl;
            }
            catch (std::exception const &e) {
                std::cout << "Caught asynchronous STL exception: " << e.what() << std::endl;
            }
        }
    };

    sycl::device dev;
    sycl::queue q;
    try {
        dev = sycl::device(selector);
        q = sycl::queue(dev, exception_handler);

        try {
            if constexpr (strict) {
                if (dev.is_cpu() || dev.is_gpu()) { //Only CPU and GPU not host, dsp, fpga, ?...
                    q.template single_task([]() {}).wait();
                }
            }
        }
        catch (...) {
            std::cerr << "Warning: " << dev.get_info<sycl::info::device::name>() << " found but not working! Fall back on: ";
            dev = sycl::device(sycl::host_selector());
            q = sycl::queue(dev, exception_handler);
            std::cerr << dev.get_info<sycl::info::device::name>() << '\n';
            return q;
        }
    }
    catch (...) {

        dev = sycl::device(sycl::host_selector());
        q = sycl::queue(dev, exception_handler);
        std::cerr << "Warning: Expected device not found! Fall back on: " << dev.get_info<sycl::info::device::name>() << '\n';
    }
    return q;
}
