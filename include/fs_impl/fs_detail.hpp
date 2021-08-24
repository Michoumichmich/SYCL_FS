#pragma once

#include <sycl/sycl.hpp>
#include <cstdio>
#include <string>
#include <async_rpc.hpp>

#ifdef IMAGE_LOAD_SUPPORT

#include "codecs/image_decoder.h"

#endif

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
        template<class ... args>
        struct nothing_matched : std::false_type {
        };

        template<class ... args>
        constexpr void fail_to_compile() {
            static_assert(nothing_matched<args...>::value);
        }


        /**
         * Returns the length of a string. Principally used to get the filename length
         * to copy it to the host
         * @param str
         * @return
         */
        static inline size_t strlen(const char *str) {
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
        static inline int memcmp(const T *str_1, const T *str_2, size_t count) {
            const auto *s_1 = (const uint8_t *) str_1;
            const auto *s_2 = (const uint8_t *) str_2;
            while (count-- > 0) {
                if (*s_1++ != *s_2++)
                    return s_1[-1] < s_2[-1] ? -1 : 1;
            }
            return 0;
        }

        template<typename T>
        static inline volatile T *memcpy_work_group(sycl::nd_item<1> item, volatile T *dst, const T *src, size_t elt_count) {
            const size_t work_item_id = item.get_local_linear_id();
            const size_t work_group_size = item.get_local_range(0);
            // Packed memory copy
            for (size_t i = work_item_id; i < elt_count; i += work_group_size) {
                dst[i] = src[i];
            }
            return dst;
        }

        template<typename T>
        static inline T *memcpy_work_group(sycl::nd_item<1> item, T *dst, const volatile T *src, size_t elt_count) {
            const size_t work_item_id = item.get_local_linear_id();
            const size_t work_group_size = item.get_local_range(0);
            // Packed memory copy
            for (size_t i = work_item_id; i < elt_count; i += work_group_size) {
                dst[i] = src[i];
            }
            return dst;
        }

        template<typename T>
        static inline volatile T *memcpy(volatile T *dst, const T *src, size_t elt_count) {
            for (size_t i = 0; i < elt_count; ++i) {
                dst[i] = src[i];
            }
            return dst;
        }

        template<typename T>
        static inline T *memcpy(T *dst, const volatile T *src, size_t elt_count) {
            for (size_t i = 0; i < elt_count; ++i) {
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
            write,
#ifdef IMAGE_LOAD_SUPPORT
            load_image
#endif //IMAGE_LOAD_SUPPORT
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
            FILE *fd = nullptr;
            [[maybe_unused]] size_t pad_2 = 0;
        };

        static open_return open(const open_args &args) {
            //std::cerr << "[info] Opening: " << std::string(args.filename) << std::endl;
            switch (args.opening_mode) {
                case fs_mode::read_only:
                    return open_return{.fd =fopen(args.filename, "rb")};
                case fs_mode::write_only:
                    return open_return{.fd =fopen(args.filename, "wb")};
                case fs_mode::append_only:
                    return open_return{.fd =fopen(args.filename, "ab")};
                case fs_mode::read_write:
                    return open_return{.fd =fopen(args.filename, "rb+")};
                case fs_mode::erase_read_write:
                    return open_return{.fd =fopen(args.filename, "wb+")};
                case fs_mode::append_read:
                    return open_return{.fd =fopen(args.filename, "ab+")};
                case fs_mode::none:
                default:
                    return open_return{.fd = nullptr};
            }


        }

        /**
         * Handling of the closing of a file
         */
        struct close_args {
            [[maybe_unused]] size_t pad_1 = 0;
            FILE *fd;
            [[maybe_unused]] size_t pad_2 = 0;
        };

        static inline void close(const close_args &args) {
            if (args.fd) {
                fclose(args.fd);
            }
        }


        /**
         * Handles reading from a file
         */
        struct read_args {
            [[maybe_unused]] int32_t pad_1 = 0;
            volatile int32_t offset = 0;
            volatile fs_offset offset_type = fs_offset::current;
            volatile void *volatile ptr = nullptr; // Volatile pointer to volatile data (yes)
            volatile size_t size_bytes_elt = 1;
            volatile size_t elt_count = 1;
            FILE *volatile fd = nullptr;
        };

        struct read_return {
            [[maybe_unused]] size_t pad_1 = 0;
            size_t bytes_read = 0;
            [[maybe_unused]] size_t pad_2 = 0;
        };

        static inline read_return read(const read_args &args) {
            if (args.offset_type != fs_offset::current || args.offset != 0) {
                fseek(args.fd, args.offset * (int32_t) args.size_bytes_elt, (enum_storage_t) args.offset_type);
            }
            return read_return{.bytes_read = args.size_bytes_elt * args.elt_count * fread((void *) args.ptr, args.size_bytes_elt * args.elt_count, 1, args.fd)};
        }

        /**
         * Handles writing to a file
         */
        struct write_args {
            [[maybe_unused]] int32_t pad_1 = 0;
            volatile int32_t offset = 0;
            volatile fs_offset offset_type = fs_offset::current;
            const volatile void *volatile ptr = nullptr; // Read only volatile pointer to volatile data
            volatile size_t size_bytes_elt = 0;
            volatile size_t elt_count = 0;
            FILE *volatile fd = nullptr;
        };

        struct write_return {
            [[maybe_unused]] size_t pad_1 = 0;
            size_t bytes_written = 0;
            [[maybe_unused]] size_t pad_2 = 0;
        };

        static inline write_return write(const write_args &args) {
            if (args.offset_type != fs_offset::current || args.offset != 0) {
                fseek(args.fd, args.offset * (int32_t) args.size_bytes_elt, (enum_storage_t) args.offset_type);
            }
            return write_return{.bytes_written= args.size_bytes_elt * args.elt_count * fwrite((void *) args.ptr, args.size_bytes_elt * args.elt_count, 1, args.fd)};
        }

#ifdef IMAGE_LOAD_SUPPORT
        /**
         * Handles reading an image
         */
        struct image_reading_args {
            [[maybe_unused]] size_t pad_1 = 0;
            size_t available_space = 0;
            char filename[fs_max_filename_len] = {0};
            char *ptr = nullptr;
            [[maybe_unused]] size_t pad_2 = 0;
        };

        struct image_reading_return {
            [[maybe_unused]] size_t pad_1 = 0;
            size_t x = 0;
            size_t y = 0;
            [[maybe_unused]] size_t pad_2 = 0;
        };

        static inline image_reading_return read_image(const image_reading_args &args) {
            using namespace std::string_literals;
            int x, y, n, ok;
            ok = stbi_info(args.filename, &x, &y, &n);
            if (!ok) {
                throw std::runtime_error("Failed opening: "s + args.filename);
            } else if (sizeof(char) * x * y * 4 > args.available_space) {
                throw std::runtime_error("Buffer too small to load: "s + args.filename);
            }

            unsigned char *data = stbi_load(args.filename, &x, &y, &n, 4);
            std::memcpy(args.ptr, data, sizeof(char) * x * y * 4);
            return image_reading_return{.x=(size_t) x, .y=(size_t) y};
        }

#endif //IMAGE_LOAD_SUPPORT

        /**
         * Union containing the input arguments
         */
        union fs_args {
            struct open_args open_{};
            struct close_args close_;
            struct read_args read_;
            struct write_args write_;
#ifdef IMAGE_LOAD_SUPPORT
            struct image_reading_args load_image_;
#endif //IMAGE_LOAD_SUPPORT
        };

        /**
         * Union containing the output return value(s)
         */
        union fs_returns {
            struct open_return open_{};
            struct read_return read_;
            struct write_return write_;
#ifdef IMAGE_LOAD_SUPPORT
            struct image_reading_return load_image_;
#endif //IMAGE_LOAD_SUPPORT
        };

        /**
         * Function called by the HOST in a loop, on runnable Channels. Here we do/undo our unions, etc.
         * @param in pointer to a "channel" that contains data from the SYCL kernel RPC call and where we will
         * put the return value. If we want to read/write files asynchronously on the host too, it's here that
         * is should be done.
         */
        template<bool use_dma, bool use_pinned_memory>
        static inline void runner_function(sycl::rpc::rpc_channel<functions_def, fs_args, fs_returns> *in) {
            if constexpr(use_dma && use_pinned_memory) {
                fail_to_compile<use_dma>();
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
#ifdef IMAGE_LOAD_SUPPORT
                case functions_def::load_image:
                    in->set_retval(fs_returns{.load_image_ = read_image(in->get_func_args().load_image_)});
                    break;
#endif //IMAGE_LOAD_SUPPORT

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