#pragma once

#include "common.h"

namespace sycl {

    template<typename T, bool use_dma, bool use_pinned_memory>
    class fs_accessor_work_group : protected fs_accessor<T, use_dma, use_pinned_memory> {
        friend fs<T, true, use_dma, use_pinned_memory>;
        friend fs<T, false, use_dma, use_pinned_memory>;
    private:
        fs_detail::local_accessor_fs_descriptor_work local_mem_;
    protected:
        fs_accessor_work_group(sycl::handler& cgh, rpc_accessor_t accessor, size_t channel_count, size_t buffer_len, T* buffers)
                :fs_accessor<T, use_dma, use_pinned_memory>(accessor, channel_count, buffer_len, buffers),
                 local_mem_(fs_detail::local_accessor_fs_descriptor_work(sycl::range<1>(1), cgh))
        {
        }

    public:
        using fs_accessor<T, use_dma, use_pinned_memory>::get_channel_count;

        template<fs_mode mode>
        std::optional<fs_descriptor_work_group < T, mode, use_dma, use_pinned_memory>> open(sycl::nd_item<1> item, size_t channel_idx, const char* filename) const
        {
            if (item.get_local_linear_id() == 0) {
                if (channel_idx >= this->channel_count_) {
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
                if (!this->accessor_.template call_remote_procedure<fs_detail::functions_def::open>(channel_idx, args, true)) {
                    return std::nullopt;
                }
                struct fs_detail::open_return res = this->accessor_.get_result(channel_idx).open_;
                this->accessor_.release(channel_idx);
                local_mem_[0].open_v = res;
            }

            /**
             * One thread opened the file and saved the result in the local memory,
             * we then
             */
            item.barrier(sycl::access::fence_space::local_space);
            if (!local_mem_[0].open_v.fd) {
                return std::nullopt;
            }
            else {
                return fs_descriptor_work_group<T, mode, use_dma, use_pinned_memory>(item, local_mem_, this->accessor_, channel_idx, local_mem_[0].open_v, this->get_host_buffer(channel_idx),
                        this->buffer_len_);
            }

        }
    };
}