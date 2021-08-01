#pragma once

#include "common.h"
#include "fs_descriptor.hpp"

namespace sycl {

    template<typename T, fs_mode open_mode, bool use_dma, bool use_pinned_memory>
    class fs_descriptor_work_group {
        friend fs_accessor_work_group<T, use_dma, use_pinned_memory>;
    private:
        fs_descriptor<T, open_mode, use_dma, use_pinned_memory> base_descriptor_;
        sycl::nd_item<1> item_;
        fs_detail::local_accessor_fs_descriptor_work local_mem_;
    protected:
        fs_descriptor_work_group(const sycl::nd_item<1>& item,
                const fs_detail::local_accessor_fs_descriptor_work& local_mem,
                rpc_accessor_t acc,
                size_t channel_idx,
                const struct fs_detail::open_return& open,
                T* host_buffer,
                size_t buffer_len)
                :base_descriptor_(acc, channel_idx, open, host_buffer, buffer_len),
                 item_(item),
                 local_mem_(local_mem) { }

    public:

        size_t write(const T* device_src, size_t elt_count, fs_offset offset_type = fs_offset::current, int32_t file_offset = 0)
        {
            const size_t work_item_id = item_.get_local_linear_id();
            if (!base_descriptor_.open_v_.fd) {
                return 0;
            }
            if constexpr (open_mode == fs_mode::read_only) {
                return 0;
            }
            if (elt_count > base_descriptor_.buffer_len_) {
                return 0;
            }
            // We acquire the channel which guarantees we're the only ones using the host_buffer which is tied to the channel index

            struct fs_detail::write_args args{};
            if (work_item_id == 0) {
                if (!base_descriptor_.accessor_.acquire(base_descriptor_.channel_idx_)) {
                    local_mem_[0].was_acquired = false;
                    local_mem_[0].retval = 0;
                    return 0;
                }
                local_mem_[0].was_acquired = true;
                args.fd = base_descriptor_.open_v_.fd;
                args.elt_count = elt_count;
                args.size_bytes_elt = sizeof(T);

                if constexpr(use_dma) {
                    args.ptr = (const void*) device_src;
                }
                else {
                    args.ptr = base_descriptor_.host_buffer_;
                }

                args.offset = file_offset;
                args.offset_type = offset_type;
            }

            item_.barrier(sycl::access::fence_space::local_space);

            if constexpr (!use_dma) {
                if (local_mem_[0].was_acquired) {
                    fs_detail::memcpy_work_group<T>(item_, base_descriptor_.host_buffer_, device_src, elt_count);
                }
                item_.barrier(sycl::access::fence_space::local_space);
            }

            if (work_item_id == 0) {
                // Doing the call
                bool spawn = (sizeof(T) * elt_count) > byte_threshold;
                base_descriptor_.accessor_.template call_remote_procedure<fs_detail::functions_def::write, false>(base_descriptor_.channel_idx_, fs_detail::fs_args{.write_ = args}, spawn);
                auto result = base_descriptor_.accessor_.get_result(base_descriptor_.channel_idx_);
                local_mem_[0].retval = result.write_.bytes_written / sizeof(T);
                //printf("bytes %lu \n",result.write_v.bytes_written);
                base_descriptor_.accessor_.release(base_descriptor_.channel_idx_);
            }
            item_.barrier(sycl::access::fence_space::local_space);
            return local_mem_[0].retval;
        }

        size_t read(T* device_ptr, size_t elt_count, fs_offset offset_type = fs_offset::current, int32_t offset = 0)
        {
            const size_t work_item_id = item_.get_local_linear_id();
            if (!base_descriptor_.open_v_.fd) {
                return 0;
            }

            if constexpr (open_mode == fs_mode::write_only || open_mode == fs_mode::append_only) {
                return 0;
            }

            if (elt_count > base_descriptor_.buffer_len_) {
                return 0;
            }
            // We acquire the channel which guarantees we're the only ones using the host_buffer which is tied to the channel index

            struct fs_detail::read_args args{};
            if (work_item_id == 0) {
                if (!base_descriptor_.accessor_.acquire(base_descriptor_.channel_idx_)) {
                    local_mem_[0].was_acquired = false;
                    local_mem_[0].retval = 0;
                    return 0;
                }
                local_mem_[0].was_acquired = true;
                args.fd = base_descriptor_.open_v_.fd;
                args.elt_count = elt_count;
                args.size_bytes_elt = sizeof(T);

                if constexpr(use_dma) {
                    args.ptr = (void*) device_ptr;
                }
                else {
                    args.ptr = base_descriptor_.host_buffer_;
                }

                args.offset = offset;
                args.offset_type = offset_type;
                // Doing the call
                base_descriptor_.accessor_.template call_remote_procedure<fs_detail::functions_def::read, false>(base_descriptor_.channel_idx_, fs_detail::fs_args{.read_ = args}, true);
                local_mem_[0].retval = base_descriptor_.accessor_.get_result(base_descriptor_.channel_idx_).read_.bytes_read / sizeof(T);
            }
            item_.barrier(sycl::access::fence_space::local_space);

            if constexpr(!use_dma) {
                if (local_mem_[0].was_acquired && local_mem_[0].retval > 0) {
                    fs_detail::memcpy_work_group<T>(item_, device_ptr, base_descriptor_.host_buffer_, elt_count);
                }
                item_.barrier(sycl::access::fence_space::local_space);
            }

            if (work_item_id == 0) {
                //printf("bytes %lu \n",result.write_v.bytes_written);
                base_descriptor_.accessor_.release(base_descriptor_.channel_idx_);
            }
            item_.barrier(sycl::access::fence_space::local_space);
            return local_mem_[0].retval;
        }

        template<template<typename, int, sycl::access_mode, sycl::target, access::placeholder> class accessor_arg, access_mode access_mode, sycl::target target, access::placeholder placeholder>
        size_t
        write(const accessor_arg<T, 1, access_mode, target, placeholder>& src_accessor, size_t accessor_begin, size_t elt_count, fs_offset offset_type = fs_offset::current, int32_t file_offset = 0)
        {
            const T* ptr = src_accessor.get_pointer() + (ptrdiff_t) accessor_begin;
            size_t accessor_size = src_accessor.get_count();
            if (accessor_begin >= accessor_size) return 0;
            size_t space_left = accessor_size - accessor_begin;
            return write(ptr, sycl::min(space_left, elt_count), offset_type, file_offset);
        }

        template<template<typename, int, sycl::access_mode, sycl::target, access::placeholder> class accessor_arg, access_mode access_mode, sycl::target target, access::placeholder placeholder>
        size_t read(accessor_arg<T, 1, access_mode, target, placeholder>& dst_accessor, size_t accessor_begin, size_t elt_count, fs_offset offset_type = fs_offset::current, int32_t file_offset = 0)
        {
            T* ptr = dst_accessor.get_pointer() + (ptrdiff_t) accessor_begin;
            size_t accessor_size = dst_accessor.get_count();
            if (accessor_begin >= accessor_size) return 0;
            size_t space_left = accessor_size - accessor_begin;
            return read(ptr, sycl::min(space_left, elt_count), offset_type, file_offset);
        }

        void close()
        {
            if (item_.get_local_linear_id() == 0) {
                base_descriptor_.close();
            }
        }

        /**
         * Queries the file descriptor to get the maximum number of elts T one could read/write at once.
         * @return
         */
        size_t get_max_single_io_count()
        {
            return base_descriptor_.get_max_single_io_count();
        }

    };
}