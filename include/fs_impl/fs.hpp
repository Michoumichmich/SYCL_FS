#pragma once

#include "common.h"

namespace sycl {

    /**
     * @class fs
     * Created on the host.
     * It should at lease outlive the kernel execution time.
     * @tparam T type of elements to read/write
     * @tparam parallel_host_file_io whether we want the HOST to access files in parallel and spawning threads
     */
    template<typename T, bool parallel_host_file_io, bool use_dma, bool use_pinned_memory>
    class fs {

    private:
        sycl::queue q_;
        size_t channel_count_;
        size_t buffer_len_;
        sycl::async_rpc<fs_detail::functions_def, fs_detail::fs_args, fs_detail::fs_returns, parallel_host_file_io> rpc_runner_;
        T *buffers_;

    public:

        /**
         * Creates a fs on the host which initializes the file api.
         * @param q the queue on which we want to use the fs api
         * @param channel_count number of parallel channels to open
         * @param buffer_len maximum length of a buffer of type T.
         */
        fs(const sycl::queue &q, size_t channel_count, size_t buffer_len, double frequency = 100000)
                : q_(q),
                  channel_count_(channel_count),
                  buffer_len_(buffer_len),
                  rpc_runner_(channel_count_, q_, fs_detail::runner_function < use_dma, use_pinned_memory > , frequency) {
            buffers_ = sycl::malloc_host<T>(channel_count_ * buffer_len_, q_);
            assert(buffers_ && "Cannot allocate file buffers on host");
        };

        /**
         * Returns an accessor to be used by the kernel
         */
        fs_accessor <T, use_dma, use_pinned_memory> get_access() const {
            return fs_accessor<T, use_dma, use_pinned_memory>(rpc_runner_.template get_access<true>(), channel_count_, buffer_len_, buffers_);
        }

        fs_accessor_work_group <T, use_dma, use_pinned_memory> get_access_work_group(sycl::handler &cgh) const {
            return fs_accessor_work_group<T, use_dma, use_pinned_memory>(cgh, rpc_runner_.template get_access<true>(), channel_count_, buffer_len_, buffers_);
        }

        fs(const fs &) = delete;

        fs(fs &&) noexcept = default;

        fs &operator=(fs &&) noexcept = default;

        ~fs() {
            sycl::free(buffers_, q_);
        }

        /**
         * Returns the free memory (in bytes) needed on the host in order for the API to work on the `sycl::queue` q.
         */
        static size_t required_host_alloc_size(const sycl::queue &q, size_t channel_count, size_t buffer_len) {
            (void) q;
            size_t channel_alloc_size = sycl::async_rpc<fs_detail::functions_def, fs_detail::fs_args, fs_detail::fs_returns, parallel_host_file_io>::required_alloc_size(channel_count);
            if constexpr(use_dma) {
                return channel_alloc_size;
            } else {
                return channel_count * buffer_len * sizeof(T) + channel_alloc_size;
            }
        }

        /**
         * Returns the free memory (in bytes) needed on the device local memory in order for the API to work on the `sycl::queue` q.
         */
        static size_t required_local_alloc_size_work_group(const sycl::queue &q, size_t channel_count, size_t buffer_len) {
            (void) channel_count;
            (void) q;
            (void) buffer_len;
            return sizeof(fs_detail::fs_accessor_local_mem);
        }

        static size_t required_device_alloc_size(const sycl::queue &q, size_t channel_count, size_t buffer_len) {
            (void) q;
            if constexpr (use_dma && use_pinned_memory) {
                return channel_count * buffer_len * sizeof(T);
            } else {
                return 0;
            }
        }

        /**
         * Returns whether the queue meets the required capabilities?
         */
        static bool has_support(const sycl::queue &q) {
            return sycl::async_rpc<fs_detail::functions_def, fs_detail::fs_args, fs_detail::fs_returns, parallel_host_file_io>::has_support(q);
        }

        static bool has_dma(const sycl::queue &q, const std::string &file = {}) {
            (void) file;
            return q.get_device().is_cpu() || q.get_device().is_host();
        }


    };

}