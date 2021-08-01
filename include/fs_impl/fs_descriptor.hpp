#pragma once

#include "common.h"

namespace sycl {
    /**
   * File descriptor returned by a fs_accessor after opening a file.
   * This descriptor holds the data needed to do communication.
   * We can notice it is bound to the original channel. This could be changed
   * to dynamically change of channel, to find a free one.
   * @tparam T Type of element read
   * @tparam open_mode Mode of file opening chosen when calling accessor's open.
   */
    template<typename T, fs_mode open_mode, bool use_dma, bool use_pinned_memory>
    class fs_descriptor {
        friend fs_descriptor_work_group<T, open_mode, use_dma, use_pinned_memory>;
        friend fs_accessor<T, use_dma, use_pinned_memory>;
    protected:
        const rpc_accessor_t accessor_;
        const size_t channel_idx_;
        struct fs_detail::open_return open_v_;
        T* host_buffer_;
        const size_t buffer_len_;

        /**
         * Shouldn't be called by a user.
         */
        fs_descriptor(rpc_accessor_t acc, size_t channel_idx, const struct fs_detail::open_return& open, T* host_buffer, size_t buffer_len)
                :
                accessor_(acc),
                channel_idx_(channel_idx),
                open_v_(open),
                host_buffer_(host_buffer),
                buffer_len_(buffer_len)
        {
        }

    public:

        /**
         * Write elt_count elements of type T, from device_ptr to the file.
         * @param device_ptr pointer to the elements of type T.
         * @param elt_count number of elements of type T to write. It should be smaller than the buffer length.
         * @param offset Writing offset in elements
         * @return The number of elements written, and 0 if we couldn't open the channel, the mode does not allow writing or we want to write too many elements
         */
        size_t write(const T* device_ptr, size_t elt_count, fs_offset offset_type = fs_offset::current, int32_t offset = 0)
        {
            if (!open_v_.fd) {
                return 0;
            }

            if constexpr (open_mode == fs_mode::read_only) {
                return 0;
            }

            if (elt_count > buffer_len_) {
                return 0;
            }
            // We acquire the channel which guarantees we're the only ones using the host_buffer which is tied to the channel index
            if (!accessor_.acquire(channel_idx_)) {
                return 0;
            }

            // Initialising the arguments
            struct fs_detail::write_args args{};
            args.fd = open_v_.fd;
            args.elt_count = elt_count;
            args.size_bytes_elt = sizeof(T);
            args.ptr = host_buffer_;
            args.offset = offset;
            args.offset_type = offset_type;

            // Sending the data to the host
            memcpy(host_buffer_, device_ptr, sizeof(T) * elt_count);

            bool spawn = (sizeof(T) * elt_count) > byte_threshold;

            // Doing the call
            accessor_.template call_remote_procedure<fs_detail::functions_def::write, false>(channel_idx_, fs_detail::fs_args{.write_ = args}, spawn);
            auto result = accessor_.get_result(channel_idx_);
            size_t elts_written = result.write_.bytes_written / sizeof(T);
            //printf("bytes %lu \n",result.write_v.bytes_written);
            accessor_.release(channel_idx_);
            return elts_written;
        }

        /**
         * Read elt_count elements of type T, from the the file to dst.
         * @param dst pointer to the device memory where to store the read elements of type T.
         * @param elt_count number of elements of type T to write. It should be smaller than the buffer length.
         * @param offset Reading offset in elements
         * @return The number of elements read, and 0 if we couldn't open the channel, the mode does not allow reading or we want to read too many elements
         */
        size_t read(T* dst, size_t elt_count, fs_offset offset_type = fs_offset::current, int32_t offset = 0)
        {
            if (!open_v_.fd) {
                return 0;
            }

            if constexpr (open_mode == fs_mode::write_only || open_mode == fs_mode::append_only) {
                return 0;
            }

            if (elt_count > buffer_len_) {
                return 0;
            }

            // We acquire the channel which guarantees we're the only ones using the host_buffer which is tied to the channel index
            if (!accessor_.acquire(channel_idx_)) {
                return 0;
            }

            // Initialising the arguments
            struct fs_detail::read_args args{};
            args.fd = open_v_.fd;
            args.elt_count = elt_count;
            args.size_bytes_elt = sizeof(T);
            args.ptr = host_buffer_;
            args.offset = offset;
            args.offset_type = offset_type;

            // Doing the call
            accessor_.template call_remote_procedure<fs_detail::functions_def::read, false>(channel_idx_, fs_detail::fs_args{.read_ = args}, true);

            auto result = accessor_.get_result(channel_idx_);
            size_t elts_read = result.read_.bytes_read / sizeof(T);

            // Getting the data from the host
            memcpy(dst, host_buffer_, sizeof(T) * elt_count);

            accessor_.release(channel_idx_);
            return elts_read;
        }

        template<template<typename, int, sycl::access_mode, sycl::target, access::placeholder> class accessor_arg, access_mode access_mode, sycl::target tgt, access::placeholder pl>
        size_t write(const accessor_arg<T, 1, access_mode, tgt, pl>& accessor_arg_1, size_t accessor_begin, size_t count, fs_offset offset_type = fs_offset::current, int32_t offset = 0)
        {
            const T* ptr = accessor_arg_1.get_pointer() + (ptrdiff_t) accessor_begin;
            size_t accessor_size = accessor_arg_1.get_count();
            if (accessor_begin >= accessor_size) return 0;
            size_t space_left = accessor_size - accessor_begin;
            return write(ptr, sycl::min(space_left, count), offset_type, offset);
        }

        template<template<typename, int, sycl::access_mode, sycl::target, access::placeholder> class accessor_arg, access_mode access_mode, sycl::target tgt, access::placeholder pl>
        size_t read(accessor_arg<T, 1, access_mode, tgt, pl>& accessor_arg_1, size_t accessor_begin, size_t count, fs_offset offset_type = fs_offset::current, int32_t offset = 0)
        {
            T* ptr = accessor_arg_1.get_pointer() + (ptrdiff_t) accessor_begin;
            size_t accessor_size = accessor_arg_1.get_count();
            if (accessor_begin >= accessor_size) return 0;
            size_t space_left = accessor_size - accessor_begin;
            return read(ptr, sycl::min(space_left, count), offset_type, offset);
        }

        /**
         * Closes the file. The fs_descriptor shouldn't be used after that.
         */
        void close()
        {
            if (!open_v_.fd) { return; }
            struct fs_detail::close_args args{.fd = open_v_.fd};
            accessor_.template call_remote_procedure<fs_detail::functions_def::close>(channel_idx_, fs_detail::fs_args{.close_ = args}, true);
            accessor_.wait(channel_idx_);
            open_v_.fd = nullptr;
            accessor_.release(channel_idx_);
        }

        /**
         * Queries the file descriptor to get the maximum number of elts T one could read/write at once.
         * @return
         */
        size_t get_max_single_io_count()
        {
            return buffer_len_;
        }


    };
}