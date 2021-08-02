#include <sycl/sycl.hpp>
#include <sycl_fs.hpp>
#include <tools/sycl_queue_helpers.hpp>

class demo_fs_kernel;

int main() {
    sycl::queue q = try_get_queue(sycl::gpu_selector{}); // Change your device
    std::cout << "Running on: " << q.get_device().get_info<sycl::info::device::name>() << std::endl;
    sycl::fs<char> fs(q, 10, 200);

    q.submit([&](sycl::handler &cgh) {
        sycl::fs_accessor<char> acc = fs.get_access();
        sycl::stream os(1024, 256, cgh);
        cgh.single_task<demo_fs_kernel>([=]() {
            // Setting up some strings and data
            size_t idx = 0;
            const char file_name[] = "my_file.txt";
            const char greetings[] = "Hello, World!\nWriting this to a file from a SYCL kernel\n";
            const char success_message[] = "Success!\n";
            char tmp[sizeof(greetings)];
            size_t message_size = sycl::fs_detail::strlen(greetings); // = sizeof(greetings) - 1

            /**
             * Trying to open the file in write_only (w) mode. `fh` is a `std::optional which on success
             * is the file handle we can use to do our IO.
             */
            if (auto fh = acc.open<sycl::fs_mode::write_only>(idx, file_name)) {
                size_t written = fh->write(greetings, message_size); // Returns also the number of elements written
                if (written != message_size) {
                    os << "Written " << written << " elements instead of " << message_size << sycl::endl;
                }
                fh->close(); // Don't forget to close the file
            }

            /**
             * Reading back the message to a buffer
             */
            if (auto fh = acc.open<sycl::fs_mode::read_only>(idx, file_name)) {
                if (fh->read(tmp, message_size) != message_size) {
                    os << "Read not enough bytes" << sycl::endl;
                }
                fh->close();
            }


            /**
             * Checking whether it was read and written successfully, if so appending "Success!" to the file
             */
            if (sycl::fs_detail::memcmp(greetings, tmp, message_size) == 0) {
                if (auto fh = acc.open<sycl::fs_mode::append_only>(idx, file_name)) {
                    fh->write(success_message, sizeof(success_message) - 1);
                    fh->close();
                    os << success_message << sycl::endl;
                }
            }
        });
    }).wait();

}

