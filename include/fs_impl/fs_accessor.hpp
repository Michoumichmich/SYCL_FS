#pragma once

#include "common.h"

namespace sycl {

    /**
     * Accessor we use to open files.
     * @tparam T
     */
    template<typename T, bool use_dma, bool use_pinned_memory>
    class fs_accessor {
        friend fs<T, true, use_dma, use_pinned_memory>;
        friend fs<T, false, use_dma, use_pinned_memory>;
    protected:
        rpc_accessor_t rpc_accessor_;
        size_t channel_count_;
        size_t buffer_len_;
        T *buffers_;

    protected:
        /**
         * Shouldn't be called by the user
         * @param accessor The RPC accessor that was created on the host. It allows to call host functions.
         * @param channel_count Number of communication channels opened on the host
         * @param buffer_len Maximum length (count of elements T) of one read/write
         * @param buffers Pointers to all IO buffers
         */
        fs_accessor(rpc_accessor_t accessor, size_t channel_count, size_t buffer_len, T *buffers)
                : rpc_accessor_(accessor),
                  channel_count_(channel_count),
                  buffer_len_(buffer_len),
                  buffers_(buffers) {}

    protected:
        [[nodiscard]] T *get_host_buffer(size_t channel_idx) const {
            return buffers_ + channel_idx * buffer_len_;
        }

    public:

        /**
         * Returns the number of channel that can be opened.
         * @return
         */
        [[nodiscard]] size_t get_channel_count() const {
            return channel_count_;
        }

        /**
         * Produces a call to abort_host() on the host after it completed all pending operations.
         */
        void abort_host() const {
            return rpc_accessor_.abort();
        }

        /**
         * Opens a file in a specific mode and returns a file descriptor bound to the chosen channel.
         * @tparam mode IO mode
         * @param channel_idx Communication channel chosen
         * @param filename name of the file to open. Max 256 chars
         * @return a std optional containing, on success, the file descriptor
         */
        template<fs_mode mode>
        [[nodiscard]] std::optional<sycl::fs_descriptor<T, mode, use_dma, use_pinned_memory>> open(size_t channel_idx, const char *filename) const {
            if (channel_idx >= channel_count_) {
                return std::nullopt;
            }

            size_t filename_len = fs_detail::strlen(filename);

            if (filename_len >= fs_max_filename_len) {
                return std::nullopt;
            }

            fs_detail::fs_args args{};
            args.open_.opening_mode = mode;
            memcpy(args.open_.filename, filename, filename_len);
            //Opening the file on the host
            if (!rpc_accessor_.call_remote_procedure<fs_detail::functions_def::open>(channel_idx, args, false)) {
                return std::nullopt;
            }

            if (auto result = rpc_accessor_.get_result(channel_idx)) {
                struct fs_detail::open_return res = result->open_;
                if (res.fd) {
                    rpc_accessor_.release(channel_idx);
                    return fs_descriptor<T, mode, use_dma, use_pinned_memory>(rpc_accessor_, channel_idx, res, get_host_buffer(channel_idx), buffer_len_);
                }
            }
            rpc_accessor_.release(channel_idx);
            return std::nullopt;
        }

#ifdef IMAGE_LOAD_SUPPORT

        /**
         * Opens a file in a specific mode and returns a file descriptor bound to the chosen channel.
         * @tparam mode IO mode
         * @param channel_idx Communication channel chosen
         * @param filename name of the file to open. Max 256 chars
         * @return a std optional containing, on success, the file descriptor
         */
        template<sycl::access::mode mode, sycl::access::target target>
        [[nodiscard]] std::optional<sycl::range<2>> load_image(size_t channel_idx, const char *filename, sycl::accessor<sycl::uchar4, 2, mode, target> image_accessor) const {
            if constexpr (mode == sycl::access::mode::read) {
                fs_detail::fail_to_compile<mode>();
            }

            if (channel_idx >= channel_count_) {
                return std::nullopt;
            }

            size_t filename_len = fs_detail::strlen(filename);

            if (filename_len >= fs_max_filename_len) {
                return std::nullopt;
            }

            fs_detail::fs_args args{};
            args.load_image_.ptr = (char *) get_host_buffer(channel_idx);
            args.load_image_.available_space = sizeof(T) * buffer_len_;
            memcpy(args.load_image_.filename, filename, filename_len);
            //Load and decode the picture on the host
            if (!rpc_accessor_.call_remote_procedure<fs_detail::functions_def::load_image, true>(channel_idx, args, true)) {
                // failed to acquire the channel
                return std::nullopt;
            }


            auto call_result = rpc_accessor_.get_result(channel_idx);
            if (!call_result) {
                rpc_accessor_.release(channel_idx);
                return std::nullopt;
            }

            struct fs_detail::image_loading_return res = call_result->load_image_;

            if (res.success != 1) {
                rpc_accessor_.release(channel_idx);
                return std::nullopt;
            }

            sycl::range<2> image_size{res.y, res.x};
            if (image_accessor.get_range().get(0) < image_size.get(0) || image_accessor.get_range().get(1) < image_size.get(1)) {
                rpc_accessor_.release(channel_idx);
                return std::nullopt;
            }

            auto data_ptr = (sycl::uchar *) get_host_buffer(channel_idx);
            for (size_t y = 0; y < res.y; ++y) { // Over the lines  y=0, x=0 -> top left
                for (size_t x = 0; x < res.x; ++x) { // Within a line
                    sycl::uchar *local_ptr = data_ptr + (y * 4 * res.x) + 4 * x;
                    image_accessor[id<2>(y, x)] = sycl::uchar4{local_ptr[0], local_ptr[1], local_ptr[2], local_ptr[3]}; // RGBA
                }
            }

            rpc_accessor_.release(channel_idx);
            return image_size;
        }

#endif //IMAGE_LOAD_SUPPORT
    };
}