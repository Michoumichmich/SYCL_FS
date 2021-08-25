#include <sycl/sycl.hpp>
#include <async_rpc.hpp>
#include <tools/sycl_queue_helpers.hpp>

/**
 * A blocking function to be called from the sycl device
 * @param j
 * @return
 */
static int cpu_func_rpc_int(size_t j) {
    std::cout << "[host] Called func 1 from work-item/group " << j << " and going to sleep" << std::endl;
    std::this_thread::sleep_for(std::chrono::seconds(1));
    std::cout << "[host wakes-up] from call to func 1 from work-item/group " << j << std::endl;
    return -(int) j;
}

/**
 * Non-blocking function to be called from the device
 * @param j
 * @return
 */
static int cpu_func_rpc_float(double j) {
    std::cout << "[host] Called func 2 from work-item/group " << j << " and going to sleep" << std::endl;
    std::this_thread::sleep_for(std::chrono::seconds(1));
    std::cout << "[host wakes-up] from call to func 2 from work-item/group " << j << std::endl;
    return 2;
}

/**
 * Boilerplate setup
 */
enum class functions_def {
    cpu_func_1,
    cpu_func_2,
};

union args {
    volatile size_t f_1;
    volatile double f_2;
};

union results {
    volatile int f_1;
    volatile int f_2;
};

void runner_function(sycl::rpc::rpc_channel<functions_def, args, results> *in) {
    switch (in->get_function()) {
        case functions_def::cpu_func_1:
            in->set_retval(results{.f_1 = cpu_func_rpc_int(in->get_func_args().f_1)});
            break;
        case functions_def::cpu_func_2:
            in->set_retval(results{.f_2 = cpu_func_rpc_float(in->get_func_args().f_2)});
            break;
    }
    in->set_result_ready();
}

class demo_rpc_kernel_async;

class demo_rpc_kernel_sync;

/**
 * Main example
 *
 * The first kernel submission shows how to perform async remote procedure calls.
 * It also uses the spawning of host threads to avoid getting blocked on a blocking call (demonstrated by the use of sleep).
 *
 * The second example shows the synchronous use and with blocking function calls, but without the threaded host execution policy thing
 * @return
 */
int main() {
    size_t n_threads = 10;
    sycl::queue q = try_get_queue(sycl::gpu_selector{}); // Change your device
    std::cout << "Running on: " << q.get_device().get_info<sycl::info::device::name>() << std::endl;

    // We create an RPC runner on the host
    sycl::async_rpc<functions_def, args, results, true> rpc_runner(n_threads, q, runner_function, 10000000);

    q.submit([&](sycl::handler &cgh) {
        // Getting an asynchronous accessor for the device
        auto async_caller = rpc_runner.get_access<true>(); // true for async
        sycl::stream os(1024, 256, cgh);
        cgh.parallel_for<demo_rpc_kernel_async>(sycl::range<1>(n_threads), [=](sycl::id<1> i) {
            size_t idx = i.get(0);

            // We call the function 1, arguments are passed in a user-defined union "args"
            args func_args{.f_1 = idx};
            async_caller.call_remote_procedure<functions_def::cpu_func_1>(idx, func_args, true);

            // We can try to get the return value in a non-blocking manner but the result will probably not be ready yet
            if (auto res = async_caller.try_get_result(idx)) { // try_get returns a std::optional
                os << "[ASYNC device] Result first try_get: " << res->f_1 << sycl::endl;
            }

            //Do other stuff in meanwhile ...

            // Force to get the result (with synchronisation)
            results res_u = async_caller.get_result(idx);
            os << "[ASYNC device] Result after get: " << res_u.f_1 << sycl::endl;

            //One could also wait
            async_caller.wait(idx);

            // And then the result is immediate
            if (auto res = async_caller.try_get_result(idx)) { //
                os << "[ASYNC device] Result try_get after wait: " << res->f_1 << sycl::endl;
            }
            
            // And we release the communication channel for the other threads
            async_caller.release(idx);
        });
    }).wait();

    q.submit([&](sycl::handler &cgh) {
        auto sync_caller = rpc_runner.get_access<false>(); // false for not async
        sycl::stream os(1024, 256, cgh);
        cgh.parallel_for<demo_rpc_kernel_sync>(sycl::range<1>(n_threads), [=](sycl::id<1> i) {
            size_t idx = i.get(0);

            // We call the function 2, arguments are passed in a user-defined union "args"
            args func_args{.f_2 = (double) idx};
            auto res = sync_caller.call_remote_procedure<functions_def::cpu_func_2>(idx, func_args, false); // Explicitly disallowing spawning remote threads

            // why not adding a barrier here.
            // DPC++ seems to be adding an implicit barrier (or is it related to the output stream ?)
            if (res) {
                os << "[sync] get_result: " << res->f_2 << sycl::endl;
            } else {
                os << "[sync] call failed" << sycl::endl;
            }

            // We don't need to (and shouldn't) release the channel
        });
    }).wait();
}

