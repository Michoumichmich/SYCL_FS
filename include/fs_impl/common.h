#pragma once

#include "async_rpc.hpp"
#include "fs_detail.hpp"
#include "optional"

namespace sycl {
    using rpc_accessor_t = sycl::rpc_accessor<fs_detail::functions_def, fs_detail::fs_args, fs_detail::fs_returns, true>;

    template<typename T, fs_mode open_mode, bool use_dma = false, bool use_pinned_memory = false>
    class fs_descriptor_work_group;

    template<typename T, fs_mode open_mode, bool use_dma = false, bool use_pinned_memory = false>
    class fs_descriptor;

    template<typename T, bool use_dma = false, bool use_pinned_memory = false>
    class fs_accessor;

    template<typename T, bool use_dma = false, bool use_pinned_memory = false>
    class fs_accessor_work_group;

    template<typename T, bool parallel_host_file_io = true, bool use_dma = false, bool use_pinned_memory = false>
    class fs;
}