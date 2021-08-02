#pragma once

#include <sycl_fs.hpp>

struct pixel {
    bool operator==(const pixel &rhs) const {
        return b == rhs.b &&
               g == rhs.g &&
               r == rhs.r;
    }

    bool operator!=(const pixel &rhs) const {
        return !(rhs == *this);
    }

    friend const sycl::stream &operator<<(const sycl::stream &os, const pixel &pixel) {
        os << "b: " << (int) pixel.b << " g: " << (int) pixel.g << " r: " << (int) pixel.r;
        return os;
    }

    uint8_t b, g, r;
};

/**
 * Colors are stores in a row major order
 * @param filename
 * @param width rows size
 * @param height column size
 * @param colors
 */


/**
 * https://en.wikipedia.org/wiki/YUV
 */
static pixel yuv_2_rgb(uint8_t y_value, uint8_t u_value, uint8_t v_value) {
    float r_tmp = (float) y_value + (1.370705f * ((float) v_value - 128));
    // or fast integer computing with a small approximation
    // rTmp = yValue + (351*(vValue-128))>>8;
    float g_tmp = (float) y_value - (0.698001f * ((float) v_value - 128)) - (0.337633f * ((float) u_value - 128));
    // gTmp = yValue - (179*(vValue-128) + 86*(uValue-128))>>8;
    float b_tmp = (float) y_value + (1.732446f * ((float) u_value - 128));
    // bTmp = yValue + (443*(uValue-128))>>8;
    return pixel{
            .b = (uint8_t) sycl::clamp((int) b_tmp, 0, 255),
            .g = (uint8_t) sycl::clamp((int) g_tmp, 0, 255),
            .r = (uint8_t) sycl::clamp((int) r_tmp, 0, 255),
    };
}

namespace bmp {
    inline static size_t get_buffer_size(size_t width, size_t height) {
        assert((4 - (width * 3) % 4) % 4 == 0 && "Wrong file size");
        return std::max(40ul, 3 * width * height);
    }

    struct bmp_headers {
        uint8_t file_header[14] = {'B', 'M', 0, 0, 0, 0, 0, 0, 0, 0, 54, 0, 0, 0};
        uint8_t info_header[40] = {40, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 24, 0};

        bmp_headers(size_t width, size_t height) {
            size_t file_size = 54 + 3 * width * height;  //w is your image width, h is image height, both int
            file_header[2] = (uint8_t) (file_size);
            file_header[3] = (uint8_t) (file_size >> 8u);
            file_header[4] = (uint8_t) (file_size >> 16u);
            file_header[5] = (uint8_t) (file_size >> 24u);
            info_header[4] = (uint8_t) (width);
            info_header[5] = (uint8_t) (width >> 8u);
            info_header[6] = (uint8_t) (width >> 16u);
            info_header[7] = (uint8_t) (width >> 24u);
            info_header[8] = (uint8_t) (height);
            info_header[9] = (uint8_t) (height >> 8u);
            info_header[10] = (uint8_t) (height >> 16u);
            info_header[11] = (uint8_t) (height >> 24u);
        }
    };

    static inline bool save_picture(size_t channel_idx, sycl::fs_accessor<uint8_t> acc, const char *filename, size_t width, size_t height, pixel *pixels) {
        bmp_headers headers(width, height);
        auto fh = acc.open<sycl::fs_mode::erase_read_write>(channel_idx, filename);
        if (fh) {
            fh->write(headers.file_header, 14);
            fh->write(headers.info_header, 40);
            if (fh->write((uint8_t *) pixels, 3 * width * height) != 3 * width * height) {
                fh->close();
                return false;
            }
            fh->close();
            return true;
        }
        return false;
    }

    template<typename fs_accessor_t>
    static inline bool save_picture_work_group(sycl::nd_item<1> item, size_t channel_idx, fs_accessor_t acc, const char *filename, size_t width, size_t height, pixel *pixels) {
        bmp_headers headers(width, height);
        auto fh = acc.template open<sycl::fs_mode::erase_read_write>(item, channel_idx, filename);
        if (fh) {
            fh->write(headers.file_header, 14);
            fh->write(headers.info_header, 40);
            if (fh->write((uint8_t *) pixels, 3 * width * height) != 3 * width * height) {
                fh->close();
                return false;
            }
            fh->close();
            return true;
        }
        return false;
    }

    static inline bool load_picture(size_t channel_idx, sycl::fs_accessor<uint8_t> acc, const char *filename, size_t width, size_t height, pixel *pixels) {
        bmp_headers from_file(width, height);
        auto fh = acc.open<sycl::fs_mode::read_only>(channel_idx, filename);
        if (fh) {
            fh->read(from_file.file_header, 14);
            fh->read(from_file.info_header, 40);
            if (fh->read((uint8_t *) pixels, 3 * width * height) != 3 * width * height) {
                fh->close();
                return false;
            }
            fh->close();
            return true;
        }
        return false;
    }

    template<typename fs_accessor_t>
    static inline bool load_picture_work_group(sycl::nd_item<1> item, size_t channel_idx, fs_accessor_t acc, const char *filename, size_t width, size_t height, pixel *pixels) {
        bmp_headers from_file(width, height);
        auto fh = acc.template open<sycl::fs_mode::read_only>(item, channel_idx, filename);
        if (fh) {
            fh->read(from_file.file_header, 14);
            fh->read(from_file.info_header, 40);
            if (fh->read((uint8_t *) pixels, 3 * width * height) != 3 * width * height) {
                fh->close();
                return false;
            }
            fh->close();
            return true;
        }
        return false;
    }


}