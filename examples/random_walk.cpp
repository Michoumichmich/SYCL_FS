/**
 * Random IO test. We generate a file conaining random offsets within that same file.
 * We read an offset, and move to that location, read the next offset, move to the other location, etc.
 *
 * Without the SYCL FS api, this would be quite "hard"/"impossible" to program if the table does not fit in memory
 * of the GPU kernel as each kernel would depend on the result of a previous kernel.
 */


#include <sycl_fs.hpp>
#include <tools/sycl_queue_helpers.hpp>
#include <random>
#include <tools/scope_chrono.hpp>

#define FILENAME "random_walk.tmp"

using index_t = int64_t;

/*******************************
 * Boilerplate setup functions *
 *******************************/


template<typename T, class ForwardIt>
static inline void do_fill_rand_on_host(ForwardIt first, ForwardIt last, T max)
{
    static std::random_device dev;
    static std::mt19937 engine(dev());
    auto generator = [&]() {
        if constexpr (std::is_integral<T>::value) {
            static std::uniform_int_distribution<T> distribution(0, max);
            return distribution(engine);
        }
        else if constexpr (std::is_floating_point<T>::value) {
            static std::uniform_real_distribution<T> distribution;
            return distribution(engine);
        }
        else if constexpr (std::is_same_v<T, sycl::half>) {
            static std::uniform_real_distribution<float> distribution;
            return distribution(engine);
        }
    };
    std::generate(first, last, generator);
}

index_t do_random_walk(const index_t* data, index_t start, size_t step_count)
{
    index_t current = start;
    for (size_t c = 0; c < step_count; ++c) {
        current = data[current];
    }
    return current;
}

std::vector<index_t> generate_file(size_t step_count, size_t size, size_t worker_count)
{
    auto* dataset = (index_t*) calloc(size, sizeof(index_t));
    do_fill_rand_on_host(dataset, dataset + size, size - 2);
    auto* f = fopen(FILENAME, "wb");
    assert(f);
    fwrite(dataset, size, sizeof(index_t), f);
    fclose(f);

    std::vector<index_t> results(worker_count);
    for (size_t i = 0; i < worker_count; ++i) {
        results[i] = do_random_walk(dataset, (index_t) i, step_count);
    }
    free(dataset);
    return results;
}

class random_walk_kernel;

/**************************************
 * The random walker that runs on GPU *
 **************************************/
std::vector<index_t> run_random_walk(sycl::queue& q, size_t step_count, size_t worker_count)
{
    std::vector<index_t> results(worker_count, 0);
    sycl::buffer<index_t, 1> res_buf(results.data(), worker_count);
    sycl::fs<index_t> fs(q, worker_count, 1, -1); // Doing a lot of small IO so we max out the frequency.
    q.submit([&](sycl::handler& cgh) {
        auto storage_accessor = fs.get_access();
        auto result_accessor = res_buf.get_access<sycl::access::mode::write>(cgh);
        cgh.parallel_for<random_walk_kernel>(sycl::range<1>(worker_count), [=](sycl::id<1> id) {
            auto data_reader = storage_accessor.open<sycl::fs_mode::read_only>(id.get(0), FILENAME);
            if (!data_reader)
                return;
            auto current = (index_t) id.get(0);
            for (size_t c = 0; c < step_count; ++c) {
                data_reader->read(&current, 1, sycl::fs_offset::begin, (int32_t) current);
            }
            data_reader->close();
            result_accessor[id.get(0)] = current;
        });
    }).wait();
    return results;
}

int main()
{
    sycl::queue q = try_get_queue(sycl::gpu_selector{});
    std::cout << "Running on: " << q.get_device().get_info<sycl::info::device::name>() << std::endl;
    assert(sycl::fs<index_t>::has_support(q) && "This queue does not support the FS api");

    size_t step_count = 1000;
    size_t size = 1024 * 1024;
    size_t worker_count = 100;
    auto expected = generate_file(step_count, size, worker_count);


    for(int c = 0 ; c < 1000 ; c++){
        scope_chrono chrono("computing values on the device");
        auto results = run_random_walk(q, step_count, worker_count);
        if (expected == results) {
            std::cout << "Success!\n";
        }
        else {
            std::cout << "Failure!\n";
        }
    }


}