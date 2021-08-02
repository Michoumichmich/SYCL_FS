#pragma once

#include <sycl/sycl.hpp>

static inline sycl::event memcpy_with_dependency(sycl::queue &q, void *dst, const void *src, size_t num_bytes, const sycl::event &dep_event) {
    return q.submit([=](sycl::handler &cgh) {
        cgh.depends_on(dep_event);
        cgh.memcpy(dst, src, num_bytes);
    });
}

static inline sycl::event memcpy_with_dependency(sycl::queue &q, void *dst, const void *src, size_t num_bytes, const std::vector<sycl::event> &dep_event) {
    return q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(dep_event);
        cgh.memcpy(dst, src, num_bytes);
    });
}

template<class T, int Dim>
using local_accessor = sycl::accessor<T, Dim, sycl::access::mode::read_write, sycl::access::target::local>;

template<class T, int Dim>
using constant_accessor = sycl::accessor<T, Dim, sycl::access::mode::read, sycl::access::target::constant_buffer>;