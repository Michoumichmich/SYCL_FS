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
        rpc_accessor_t accessor_;
        size_t channel_count_;
        size_t buffer_len_;
        T* buffers_;

    protected:
        /**
         * Shouldn't be called by the user
         * @param accessor The RPC accessor that was created on the host. It allows to call host functions.
         * @param channel_count Number of communication channels opened on the host
         * @param buffer_len Maximum length (count of elements T) of one read/write
         * @param buffers Pointers to all IO buffers
         */
        fs_accessor(rpc_accessor_t accessor, size_t channel_count, size_t buffer_len, T* buffers)
                :accessor_(accessor),
                 channel_count_(channel_count),
                 buffer_len_(buffer_len),
                 buffers_(buffers) { }

    protected:
        T* get_host_buffer(size_t channel_idx) const
        {
            return buffers_ + channel_idx * buffer_len_;
        }

    public:

        /**
         * Returns the number of channel that can be opened.
         * @return
         */
        size_t get_channel_count()
        {
            return channel_count_;
        }

        /**
         * Opens a file in a specific mode and returns a file descriptor bound to the chosen channel.
         * @tparam mode IO mode
         * @param channel_idx Communication channel chosen
         * @param filename name of the file to open. Max 256 chars
         * @return a std optional containing, on success, the file descriptor
         */
        template<fs_mode mode>
        std::optional<fs_descriptor < T, mode, use_dma, use_pinned_memory>> open(
        size_t channel_idx,
        const char* filename
        ) const
        {
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
            if (!accessor_.call_remote_procedure<fs_detail::functions_def::open>(channel_idx, args, true)) {
                return std::nullopt;
            }
            struct fs_detail::open_return res = accessor_.get_result(channel_idx).open_;
            accessor_.release(channel_idx);
            if (res.fd) {
                return fs_descriptor<T, mode, use_dma, use_pinned_memory>(accessor_, channel_idx, res, get_host_buffer(channel_idx), buffer_len_);
            }
            return std::nullopt;
        }
    };
}