#pragma once

#include <string>
#include <iostream>
#include <chrono>

/**
 * Small Chrono class that prints the time spent in a scope.
 */
class scope_chrono {
public:
    inline scope_chrono();

    inline explicit scope_chrono(std::string &&caller_name);

    inline scope_chrono(const scope_chrono &) = delete;

    scope_chrono &operator=(const scope_chrono &) = delete;

    inline double stop();

    inline ~scope_chrono();

private:
    std::string caller_;

    const std::chrono::time_point<std::chrono::high_resolution_clock, std::chrono::duration<long int, std::ratio<1, 1000000000>>> start_;
};

inline scope_chrono::scope_chrono()
        : start_(std::chrono::high_resolution_clock::now()) {
}

inline scope_chrono::~scope_chrono() {
    double elapsed_seconds = scope_chrono::stop();
    if (!caller_.empty()) {
        std::cerr << "time in " << caller_ << " : " << elapsed_seconds << "s" << std::endl;
    } else {
        std::cerr << "time " << elapsed_seconds << "s" << std::endl;
    }
}

inline scope_chrono::scope_chrono(std::string &&caller_name)
        : scope_chrono() {
    caller_ = caller_name;
}

inline double scope_chrono::stop() {
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start_);
    return static_cast<double>(duration.count()) / 1000000.0;
}
