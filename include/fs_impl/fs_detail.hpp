#pragma once

#include <sycl/sycl.hpp>
#include <cstdio>
#include <async_rpc.hpp>

namespace sycl {

    static constexpr int fs_max_filename_len = 256;
    static constexpr size_t byte_threshold = 8192 /** If writing more than this amount of bytes, we spawn a thread */;

    using enum_storage_t = int32_t;

    /**
     * File opening mode, mapped on the POSIX ones
     */
    enum class fs_mode : enum_storage_t {
        read_only /** R */,
        write_only /** W */,
        append_only /** A */,
        read_write /** R+ */,
        erase_read_write /** W+ */,
        append_read /** A+ */,
        none,
    };

    enum class fs_offset : enum_storage_t {
        begin /** Does the offset from the start of the file */ = SEEK_SET,
        end /** From the end */ = SEEK_END,
        current /** From the current position */ = SEEK_CUR,
    };

    namespace fs_detail {

        /**
         * A dummy struct used to abort the compilation
         * if in a  'if constexpr' ladder nothing has
         * matched
         * @tparam M
         */
        template<bool M>
        struct nothing_matched : std::false_type {
        };

        /**
         * Returns the length of a string. Principally used to get the filename length
         * to copy it to the host
         * @param str
         * @return
         */
        static inline size_t strlen(const char* str)
        {
            size_t i;
            for (i = 0; str[i] != '\0'; ++i);
            return i;
        }

        /**
         * Compare elements
         * @tparam T type of the elements to compare
         * @param count number of elements T to compare
         * @return see libc
         */
        template<typename T>
        static inline int memcmp(const T* str_1, const T* str_2, size_t count)
        {
            const auto* s_1 = (const uint8_t*) str_1;
            const auto* s_2 = (const uint8_t*) str_2;
            while (count-- > 0) {
                if (*s_1++ != *s_2++)
                    return s_1[-1] < s_2[-1] ? -1 : 1;
            }
            return 0;
        }

        template<typename T>
        static inline T* memcpy_work_group(sycl::nd_item<1> item, T* dst, const T* src, size_t elt_count)
        {
            const size_t work_item_id = item.get_local_linear_id();
            const size_t work_group_size = item.get_local_range(0);
            // Packed memory copy
            for (size_t i = work_item_id; i < elt_count; i += work_group_size) {
                dst[i] = src[i];
            }
            return dst;
        }

        /********************************
         * Setting up the ASYNC RPC API *
         ********************************/

        /**
         * Callable functions
         */
        enum class functions_def {
            open,
            close,
            read,
            write
        };

        /**
         * Handling of the opening of a file
         */
        struct open_args {
            fs_mode opening_mode = fs_mode::none;
            char filename[fs_max_filename_len] = {0};
        };

        struct open_return {
            [[maybe_unused]] size_t pad_1 = 0;
            FILE* fd = nullptr;
            [[maybe_unused]] size_t pad_2 = 0;
        };

        static open_return open(const open_args& args)
        {
            //std::cerr << "[info] Opening: " << std::string(args.filename) << std::endl;
            switch (args.opening_mode) {
            case fs_mode::read_only:
                return {.fd =fopen(args.filename, "rb")};
            case fs_mode::write_only:
                return {.fd =fopen(args.filename, "wb")};
            case fs_mode::append_only:
                return {.fd =fopen(args.filename, "ab")};
            case fs_mode::read_write:
                return {.fd =fopen(args.filename, "rb+")};
            case fs_mode::erase_read_write:
                return {.fd =fopen(args.filename, "wb+")};
            case fs_mode::append_read:
                return {.fd =fopen(args.filename, "ab+")};
            case fs_mode::none:
            default:
                return {.fd = nullptr};
            }


        }

        /**
         * Handling of the closing of a file
         */
        struct close_args {
            [[maybe_unused]] size_t pad_1 = 0;
            FILE* fd;
            [[maybe_unused]] size_t pad_2 = 0;
        };

        static inline void close(const close_args& args)
        {
            if (args.fd) {
                fclose(args.fd);
            }
        }

        /**
         * Handles reading from a file
         */
        struct read_args {
            [[maybe_unused]] int32_t pad_1 = 0;
            int32_t offset = 0;
            fs_offset offset_type = fs_offset::current;
            void* ptr = nullptr;
            size_t size_bytes_elt = 1;
            size_t elt_count = 1;
            FILE* fd = nullptr;
        };

        struct read_return {
            [[maybe_unused]] size_t pad_1 = 0;
            size_t bytes_read = 0;
            [[maybe_unused]] size_t pad_2 = 0;
        };

        static inline read_return read(const read_args& args)
        {
            if (args.offset_type != fs_offset::current || args.offset != 0) {
                fseek(args.fd, args.offset * (int32_t) args.size_bytes_elt, (enum_storage_t) args.offset_type);
            }
            return read_return{.bytes_read = args.size_bytes_elt * args.elt_count * fread(args.ptr, args.size_bytes_elt * args.elt_count, 1, args.fd)};
        }

        /**
         * Handles writing to a file
         */
        struct write_args {
            [[maybe_unused]] int32_t pad_1 = 0;
            int32_t offset = 0;
            fs_offset offset_type = fs_offset::current;
            const void* ptr = nullptr;
            size_t size_bytes_elt = 0;
            size_t elt_count = 0;
            FILE* fd = nullptr;
        };

        struct write_return {
            [[maybe_unused]] size_t pad_1 = 0;
            size_t bytes_written = 0;
            [[maybe_unused]] size_t pad_2 = 0;
        };

        static inline write_return write(const write_args& args)
        {
            if (args.offset_type != fs_offset::current || args.offset != 0) {
                fseek(args.fd, args.offset * (int32_t) args.size_bytes_elt, (enum_storage_t) args.offset_type);
            }
            return write_return{.bytes_written= args.size_bytes_elt * args.elt_count * fwrite(args.ptr, args.size_bytes_elt * args.elt_count, 1, args.fd)};
        }

        /**
         * Union containing the input arguments
         */
        union fs_args {
            struct open_args open_{};
            struct close_args close_;
            struct read_args read_;
            struct write_args write_;
        };

        /**
         * Union containing the output return value(s)
         */
        union fs_returns {
            struct open_return open_{};
            struct read_return read_;
            struct write_return write_;
        };

        /**
         * Function called by the HOST in a loop, on runnable Channels. Here we do/undo our unions, etc.
         * @param in pointer to a "channel" that contains data from the SYCL kernel RPC call and where we will
         * put the return value. If we want to read/write files asynchronously on the host too, it's here that
         * is should be done.
         */
        template<bool use_dma, bool use_pinned_memory>
        static inline void runner_function(sycl::rpc::rpc_channel<functions_def, fs_args, fs_returns>* in)
        {
            if constexpr(use_dma && use_pinned_memory) {
                static_assert(nothing_matched<use_dma>::type);
            }


            asm("":: :"memory"); // Memory barrier to be sure everything was written.
            switch (in->get_function()) {
            case functions_def::open: {
                struct open_return res = open(in->get_func_args().open_);
                //std::cerr << "[info] Opened fd: " << res.fd << std::endl;
                in->set_retval(fs_returns{.open_ = res});
            }
                break;
            case functions_def::close: {
                close(in->get_func_args().close_);
                in->set_func_args(fs_args{.close_ = {.fd=nullptr}});
                //std::cerr << "[info] Closing fd: " << in->get_func_args().close_a.fd << std::endl;
            }
                break;
            case functions_def::read: {
                //std::cout << "Reading fd: " << in->get_func_args().read_a.fd << " n_elts: " << in->get_func_args().read_a.elt_count << std::endl;
                struct read_return res = read(in->get_func_args().read_);
                in->set_retval(fs_returns{.read_ = res});
            }
                break;
            case functions_def::write: {
                //std::cout << "Writing fd: " << in->get_func_args().write_a.fd << " n_elts: " << in->get_func_args().write_a.elt_count << std::endl;
                struct write_return res = write(in->get_func_args().write_);
                in->set_retval(fs_returns{.write_ = res});
            }
                break;
            }
            asm("":: :"memory"); // Memory barrier to be sure everything was written.
            in->set_result_ready();
            asm("":: :"memory"); // Memory barrier to be sure everything was written.
        }

        template<class T, int Dim>
        using local_accessor = sycl::accessor<T, Dim, sycl::access::mode::read_write, sycl::access::target::local>;

        struct fs_accessor_local_mem {
            struct fs_detail::open_return open_v{};
            bool was_acquired{};
            size_t retval{};
        };

        using local_accessor_fs_descriptor_work = local_accessor<fs_accessor_local_mem, 1>;

    }
}