/**
@file async_rpc.hpp

Copyright 2021 Codeplay Software Ltd.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use these files except in compliance with the License.
You may obtain a copy of the License at
http://www.apache.org/licenses/LICENSE-2.0
For your convenience, a copy of the License has been included in this
repository.
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. 

@author Michel Migdal
*/



#pragma once

#include <sycl/sycl.hpp>
#include <thread>
#include <optional>
#include <atomic>
#include <execinfo.h>
#include <unistd.h>

namespace sycl {
    namespace rpc {

        struct global_state_data {
        private:
            uint64_t nano_time_stamp = (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();
            size_t abort_requested = 0;

        public:

            /**
             * HOST function, used by the runner
             */
            void update() {
                nano_time_stamp = (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();
            }

            /**
             * HOST function, used by the runner
             */
            [[nodiscard]] bool was_requested_abort() const {
                return abort_requested != 0;
            }

            /**
             * DEVICE, to be called in a kernel
             */
            void request_abort() {
                abort_requested = 1;
            }

            /**
             * DEVICE, to be called in a kernel
             */
            [[nodiscard]] uint64_t get_timestamp() const {
                return nano_time_stamp;
            }

            [[nodiscard]] static global_state_data *make_global(sycl::queue &q) {
                auto global_state = sycl::malloc_host<global_state_data>(1, q);
                return new(global_state) global_state_data();
            }

        };

        /**
         * State in which a channel can be. These states are not updated atomically
         * as it's not supported (but really it should as it's an integer)
         */
        enum class rpc_channel_state : int32_t {
            unused /** Channel free to start using it */,
            transient_state /** Data partially written */,
            ready_to_execute /** Just waiting for a runner thread to execute the request */,
            running /** Currently computing the result in sync manner */,
            result_ready /** Indicates whether the function finished running  */,
            caught_exception /** Indicates whether an exception was thrown by the runner on the host */
        };

        /**
         * Communication channel
         * @tparam functions_defined enum containing the various functions to be called.
         * @tparam func_args_u function arguments, could be in a union
         * @tparam ret_val_u function returns, could be in a union
         */
        template<typename functions_defined, typename func_args_u, typename ret_val_u>
        class rpc_channel {
        private:
            //size_t pad = 0;

            volatile rpc_channel_state state_ = rpc_channel_state::unused;
            volatile functions_defined function_{};
            func_args_u func_args_{};
            ret_val_u retval_{};
            size_t channel_id_ = 0;
            int64_t allowed_to_spawn_host_thread_ = true;
            int32_t channel_acquired_ = false;

        public:

#ifdef USING_DPCPP
#define ATOMIC_REF_NAMESPACE sycl::ext::oneapi
#else
#define ATOMIC_REF_NAMESPACE sycl
#endif

            /**
             * DEVICE function to get a channel
             * @return whether it could be acquired
             */
            [[nodiscard]] bool acquire_channel() {
                int32_t expected = false;
                ATOMIC_REF_NAMESPACE::atomic_ref<int32_t, ATOMIC_REF_NAMESPACE::memory_order::acq_rel, memory_scope::system, access::address_space::global_space> state_ref(channel_acquired_);
                if (state_ref.compare_exchange_weak(expected, true, ATOMIC_REF_NAMESPACE::memory_order::acquire, sycl::memory_scope::system)) {
                    state_ = rpc_channel_state::transient_state;
                    return true;
                }
                return false;
            }

            /**
             * It should never fail as a channel is always by the caller (so once), after acquiring it.
             * @return should always return true when properly used
             */
            bool release_channel() {
                int32_t expected = true;
                ATOMIC_REF_NAMESPACE::atomic_ref<int32_t, ATOMIC_REF_NAMESPACE::memory_order::acq_rel, memory_scope::system, access::address_space::global_space> state_ref(channel_acquired_);
                if (state_ref.compare_exchange_weak(expected, false, ATOMIC_REF_NAMESPACE::memory_order::release, sycl::memory_scope::system)) {
                    state_ = rpc_channel_state::unused;
                    return true;
                }
                //assert(false && "How did that happen? Was the channel released without being acquired or released twice?");
                return false;
            }

#undef ATOMIC_REF_NAMESPACE

            /**
             * DEVICE
             * Sets the channel as being executable
             */
            void set_executable() {
                state_ = rpc_channel_state::ready_to_execute;
#ifdef USING_DPCPP
                sycl::atomic_fence(sycl::memory_order::release, sycl::memory_scope::system);
#else
                asm("":: :"memory");
#endif

            }

            /**
             * HOST checks whether the channel can be executed
             * This should use atomic fences to ensure synchronisation with the DEVICE
             */
            [[nodiscard]] bool can_start_executing() {
#ifdef USING_DPCPP
                sycl::atomic_fence(sycl::memory_order::acquire, sycl::memory_scope::system);
#else
                asm("":: :"memory");
#endif
                return (state_ == rpc_channel_state::ready_to_execute);
            }

            void set_as_executing() {
                state_ = rpc_channel_state::running;
#ifdef USING_DPCPP
                sycl::atomic_fence(sycl::memory_order::acquire, sycl::memory_scope::system);
#else
                asm("":: :"memory");
#pragma message ("Atomic fences needed but not supported by the implementation")
#endif
            }

            /**
             * HOST
             */
            void set_result_ready() {
#ifdef USING_DPCPP
                sycl::atomic_fence(sycl::memory_order::release, sycl::memory_scope::system);
#else
                asm("":: :"memory");
#pragma message ("Atomic fences needed but not supported by the implementation")
#endif
                state_ = rpc_channel_state::result_ready;
            }

            void set_uncaught_exception() {
#ifdef USING_DPCPP
                sycl::atomic_fence(sycl::memory_order::release, sycl::memory_scope::system);
#else
                asm("":: :"memory");
#pragma message ("Atomic fences needed but not supported by the implementation")
#endif
                state_ = rpc_channel_state::caught_exception;
            }

            /**
             * DEVICE
             */
            [[nodiscard]] bool is_result_ready_or_exception() const {
#ifdef USING_DPCPP
                sycl::atomic_fence(sycl::memory_order::acquire, sycl::memory_scope::system);
#else
                asm("":: :"memory");
#pragma message ("Atomic fences needed but not supported by the implementation")
#endif
                return state_ == rpc_channel_state::result_ready || state_ == rpc_channel_state::caught_exception;
            }

            /**
             * DEVICE
             */
            [[nodiscard]] bool is_result_ready() const {
#ifdef USING_DPCPP
                sycl::atomic_fence(sycl::memory_order::acquire, sycl::memory_scope::system);
#else
                asm("":: :"memory");
#pragma message ("Atomic fences needed but not supported by the implementation")
#endif
                return state_ == rpc_channel_state::result_ready;
            }

            /**
             * DEVICE
             */
            void set_function(functions_defined f) {
                function_ = f;
            }

            /**
             * HOST
             */
            [[nodiscard]] functions_defined get_function() const {
                return function_;
            }

            /**
             * ?
             * @return
             */
            [[maybe_unused]] [[nodiscard]] size_t get_id() const {
                return channel_id_;
            }

            /**
             * HOST
             * @param channel_id
             */
            void set_id(size_t channel_id) {
                channel_id_ = channel_id;
            }

            /**
             * HOST
             * @return
             */
            [[nodiscard]] func_args_u get_func_args() const {
                return func_args_;
            }

            /**
             * DEVICE
             * @param func_args
             */
            void set_func_args(const func_args_u &func_args) {
                func_args_ = func_args;
            }

            /**
             * DEVICE
             * @return
             */
            [[nodiscard]] ret_val_u get_retval() const {
                return retval_;
            }

            /**
             * HOST
             * @param retval
             */
            void set_retval(const ret_val_u &retval) {
                retval_ = retval;
                asm("":: :"memory");
            }

            [[nodiscard]] bool can_spawn_host_thread() const {
#ifdef USING_DPCPP
                sycl::atomic_fence(sycl::memory_order::acquire, sycl::memory_scope::system);
#else
                asm("":: :"memory");
#pragma message ("Atomic fences needed but not supported by the implementation")
#endif
                return allowed_to_spawn_host_thread_;
            }

            void set_allowed_to_spawn_host_thread(bool allowed_to_spawn_host_thread) {
                allowed_to_spawn_host_thread_ = allowed_to_spawn_host_thread;
            }


        };

    }

    /**
     * All functions are executed here from the DEVICE
     * @tparam functions_defined
     * @tparam func_args_u
     * @tparam ret_val_u
     */
    template<typename functions_defined, typename func_args_u, typename ret_val_u, bool is_async>
    class rpc_accessor;

    template<typename functions_defined, typename func_args_u, typename ret_val_u>
    class rpc_accessor<functions_defined, func_args_u, ret_val_u, false> {
    private:

        /**
         * The accessor does not own the channels, so this field is mutable as it will
         * be changed by the host to establish the communication.
         */
        mutable rpc::rpc_channel<functions_defined, func_args_u, ret_val_u> *channels_;
        const size_t channel_count_;
        mutable rpc::global_state_data *global_state_;

        void wait_for_call_completion(size_t channel_idx) const {
            while (!channels_[channel_idx].is_result_ready_or_exception()) {}
        }

        void clear_channel(size_t channel_idx) const {
            channels_[channel_idx].set_retval({});
            channels_[channel_idx].set_func_args({});
        }

    public:
        /**
         * Constructor not meant to be called by the user
         * @param channels
         * @param channel_count
         */
        rpc_accessor(rpc::rpc_channel<functions_defined, func_args_u, ret_val_u> *channels, size_t channel_count, rpc::global_state_data *state)
                : channels_(channels),
                  channel_count_(channel_count),
                  global_state_(state) {}

        /**
         * In synchronous mode, the call is blocking, but automatically releases the channel and returns
         * a `std::optional` with the result, on success.
         *
         * @tparam func Function to call
         * @tparam do_acquire_channel We can ask not to acquire the channel if we pre-acquired it
         * @tparam do_release_channel We can ask not to release the channel if we want to use it after.
         * @param channel_idx Index of the channel used
         * @param args function arguments squashed in an union (but not necessarily)
         * @param can_spawn_host_thread indicates whether we allow the RPC to be launched in a separate thread on the side of the host
         * this can be useful to be set to false if we know that the function call is quick. But if we're doing ASYNC RPC and on the host, the function call is
         * blocking, it's not great. At the same time, launching inexpensive small functions in a thread can be very costly too.
         * @return the return an optional indicating whether the call succeeded.
         */
        template<functions_defined func, bool do_acquire_channel = true, bool do_release_channel = true>
        [[nodiscard]] std::optional<ret_val_u> call_remote_procedure(size_t channel_idx, const func_args_u &args, bool can_spawn_host_thread = true) const {
            if constexpr(do_acquire_channel) {
                if (channel_idx >= channel_count_ || !channels_[channel_idx].acquire_channel()) {
                    return std::nullopt; // Channel not acquired
                }
            }

            // Channel is in transient state
            channels_[channel_idx].set_function(func);
            channels_[channel_idx].set_func_args(args);
            channels_[channel_idx].set_allowed_to_spawn_host_thread(can_spawn_host_thread);
            sycl::atomic_fence(sycl::memory_order::release, sycl::memory_scope::system);
            channels_[channel_idx].set_executable();

            wait_for_call_completion(channel_idx);
            if (channels_[channel_idx].is_result_ready()) {
                auto result = channels_[channel_idx].get_retval();
                if constexpr(do_release_channel) {
                    channels_[channel_idx].release_channel();
                }
                return result;
            } else { // An exception was thrown
                return std::nullopt;
            }
        }


    public:

        /**
         * DEVICE function.
         * Will initiate the abort_host() of the running program. First all running operations will be completed before aborting.
         * There could be a delay, especially if the RPC listener is running in single threaded mode, but we don't want any
         * form of corruption.
         */
        void abort() const {
            global_state_->request_abort();
        }

        /**
         * DEVICE function. Returns the current timestamp in nanoseconds. It is updated
         * depending on the frequency chosen. Might not be as accurate as expected.
         * @return uint64_t
         */
        [[nodiscard]] uint64_t get_timestamp() const {
            return global_state_->get_timestamp();
        }

    };

    /**
     * ASYNC specialization
     */
    template<typename functions_defined, typename func_args_u, typename ret_val_u>
    class rpc_accessor<functions_defined, func_args_u, ret_val_u, true> {
    private:

        /**
         * The accessor does not own the channels, so this field is mutable as it will
         * be changed by the host to establish the communication.
         */
        mutable rpc::rpc_channel<functions_defined, func_args_u, ret_val_u> *channels_;
        const size_t channel_count_;
        mutable rpc::global_state_data *global_state_;

        inline void wait_for_call_completion(size_t channel_idx) const {
            while (!channels_[channel_idx].is_result_ready_or_exception()) {}
        }

        inline void clear_channel(size_t channel_idx) const {
            channels_[channel_idx].set_retval({});
            channels_[channel_idx].set_func_args({});
        }

    public:
        /**
         * Constructor not meant to be called by the user
         * @param channels
         * @param channel_count
         */
        rpc_accessor(rpc::rpc_channel<functions_defined, func_args_u, ret_val_u> *channels, size_t channel_count, rpc::global_state_data *state)
                : channels_(channels),
                  channel_count_(channel_count),
                  global_state_(state) {}

        /**
         * The call is non blocking, and never releases the channel.
         * @tparam func Function to call
         * @tparam do_acquire_channel We can ask not to acquire the channel if we pre-acquired it
         * @param channel_idx Index of the channel used
         * @param args function arguments squashed in an union (but not necessarily)
         * @param can_spawn_host_thread indicates whether we allow the RPC to be launched in a separate thread on the side of the host
         * this can be useful to be set to false if we know that the function call is quick. But if we're doing ASYNC RPC and on the host, the function call is
         * blocking, it's not great. At the same time, launching inexpensive small functions in a thread can be very costly too.
         * @return the return a bool indicating whether the call succeeded
         */
        template<functions_defined func, bool do_acquire_channel = true>
        [[nodiscard]] inline bool call_remote_procedure(size_t channel_idx, const func_args_u &args, bool can_spawn_host_thread = true) const {
            if constexpr(do_acquire_channel) {
                if (channel_idx >= channel_count_ || !channels_[channel_idx].acquire_channel()) {
                    return false;
                }
            }

            // Channel is in transient state
            channels_[channel_idx].set_function(func);
            channels_[channel_idx].set_func_args(args);
            channels_[channel_idx].set_allowed_to_spawn_host_thread(can_spawn_host_thread);
            sycl::atomic_fence(sycl::memory_order::release, sycl::memory_scope::system);
            channels_[channel_idx].set_executable();

            // We cannot return a std::nullopt as we don't yet know if the call has succeeded
            return true;
        }

        /**
         * DEVICE function.
         * Will initiate the abort_host() of the running program. First all running operations will be completed before aborting.
         * There could be a delay, especially if the RPC listener is running in single threaded mode, but we don't want any
         * form of corruption.
         */
        inline void abort() const {
            global_state_->request_abort();
        }

        /**
         * DEVICE function. Returns the current timestamp in nanoseconds. It is updated
         * depending on the frequency chosen. Might not be as accurate as expected.
         * @return uint64_t
         */
        [[nodiscard]] inline uint64_t get_timestamp() const {
            return global_state_->get_timestamp();
        }

        /**
         * NON-BLOCKING Returns an asynchronous result if available.
         * @param channel_idx
         * @return
         */
        [[nodiscard]] inline std::optional<ret_val_u> try_get_result(size_t channel_idx) const {
            if (channels_[channel_idx].is_result_ready()) {
                return channels_[channel_idx].get_retval();
            }
            return std::nullopt;
        }

        /**
         * BLOCKING Returns an asynchronous result. Calling if with a sync api results in UB.
         * @param channel_idx
         * @return
         */
        [[nodiscard]] inline std::optional<ret_val_u> get_result(size_t channel_idx) const {
            wait_for_call_completion(channel_idx);
            return try_get_result(channel_idx);
        }


        /**
         * ASYNC ONLY release the channel after getting your result back.
         * This increase the lifetime of the communication channel which allows
         * to perform other operations between the host and the device such as memcpy between buffers
         * that are tied to a channel_idx.
         * @param channel_idx
         */
        void inline release(size_t channel_idx) const {
            wait_for_call_completion(channel_idx);
            clear_channel(channel_idx);
            channels_[channel_idx].release_channel();
        }

        /**
         * ASYNC ONLY acquires the channel after getting your result back.
         * This increase the lifetime of the communication channel which allows
         * to perform other operations between the host and the device such as memcpy between buffers
         * that are tied to a channel_idx.
         *
         * If a channel was acquired here, we need to pass the template parameter `need_to_acquire` set to false
         * to `call_remote_procedure` so it does not try to acquire the channel.
         * @param channel_idx
         */
        [[nodiscard]] inline bool acquire(size_t channel_idx) const {
            return channels_[channel_idx].acquire_channel();
        }
    };

    template<typename functions_defined, typename func_args_u, typename ret_val_u, bool parallel_runners>
    class async_rpc {
    private:
        using rpc_channel_t = rpc::rpc_channel<functions_defined, func_args_u, ret_val_u>;
        rpc::rpc_channel<functions_defined, func_args_u, ret_val_u> *channels_ = nullptr;
        rpc::global_state_data *global_state_ = nullptr;
        sycl::queue q_;
        size_t channel_count_;
        volatile bool *keep_running_listener_ = nullptr; //DANGEROUS we're probably doing bad things by returning the reference of a local variable, but we can control the lifetime of the function so...
        std::thread listener_;

        static inline void thread_runner(
                void (*f)(rpc_channel_t *),
                rpc_channel_t *channels,
                size_t count,
                volatile bool **class_switch_address,
                double frequency,
                rpc::global_state_data *global_state) noexcept(!parallel_runners) {
            volatile bool keep_running_listener = true;
            const std::chrono::nanoseconds sleep_time = std::chrono::nanoseconds((frequency > 0) ? (int64_t) ((1e9 / frequency)) : 0);
            *class_switch_address = &keep_running_listener;
            std::vector<std::atomic<int64_t>> running_channels(count);
            for (auto &item: running_channels) {
                item = false;
            }
            std::vector<std::thread> running_threads(count);
            while (keep_running_listener) {

                /* Updating the global shared state and initiating shutdown if needed */
                global_state->update();
                if (global_state->was_requested_abort()) {
                    keep_running_listener = false;
                }

                for (size_t i = 0; i < count && keep_running_listener; ++i) {
                    int64_t expected_false_running_state = false;
                    // We check if the channel can be executed, if so then we acquire a lock
                    if (channels[i].can_start_executing() && running_channels[i].compare_exchange_strong(expected_false_running_state, true)) {
                        //We clear the thread
                        auto func = [=, &running_channels]() noexcept {
                            channels[i].set_as_executing();
                            try {
                                asm("":: :"memory"); // Memory barrier to be sure everything was written.
                                f(channels + i);
                                asm("":: :"memory"); // Memory barrier to be sure everything was written.
                                channels[i].set_result_ready();
                                asm("":: :"memory"); // Memory barrier to be sure everything was written.
                            } catch (std::exception &e) {
                                std::cerr << "Uncaught exception thrown in RPC runner (on the host).\n"
                                             "Silencing it to preserve other function calls that might be running in parallel\n"
                                             "Function call will appear as failed on the device\n"
                                             "Exception was: " <<
                                          e.what() << std::endl;
                                channels[i].set_uncaught_exception();
                            }
                            int64_t expected_after_run = true;
                            running_channels[i].compare_exchange_strong(expected_after_run, false);
                        };

                        if (parallel_runners && channels[i].can_spawn_host_thread()) {
                            if (running_threads[i].joinable()) {
                                running_threads[i].join();
                            }
                            running_threads[i] = std::thread(func);
                        } else {
                            func();
                        }
                    }
                }
                if (frequency > 0) {
                    std::this_thread::sleep_for(sleep_time); //We can slow down the exec speed of the runner to avoid having a thread go crazy
                }
            }
            // Joining threads when destructor called
            for (auto &thread: running_threads) {
                if (thread.joinable()) {
                    thread.join();
                }
            }

            // aborting if needed
            if (global_state->was_requested_abort()) {
                std::cerr << "Abort called from a thread" << std::endl;
                void *array[10];
                int size = backtrace(array, 10);
                backtrace_symbols_fd(array, size, STDERR_FILENO);
                abort();
            }


        }

    public:

        async_rpc(size_t channel_count, sycl::queue &q, void (*rpc_channel_runner)(rpc_channel_t *), double runner_frequency = -1)
                : q_(q),
                  channel_count_(channel_count) {
            assert(channel_count > 0);
            assert(rpc_channel_runner);
            channels_ = sycl::malloc_host<rpc::rpc_channel<functions_defined, func_args_u, ret_val_u>>(channel_count_, q_);
            assert(channels_ && "Allocating the channels failed on the host");
            for (size_t i = 0; i < channel_count_; ++i) {
                channels_[i] = rpc::rpc_channel<functions_defined, func_args_u, ret_val_u>{};
                channels_[i].set_id(i);
            }
            global_state_ = rpc::global_state_data::make_global(q);
            listener_ = std::thread(thread_runner, rpc_channel_runner, channels_, channel_count_, &keep_running_listener_, runner_frequency, global_state_);
        }

        /**
         *
         * @tparam async
         * @return
         */
        template<bool async>
        [[nodiscard]] rpc_accessor<functions_defined, func_args_u, ret_val_u, async> get_access() const {
            return rpc_accessor<functions_defined, func_args_u, ret_val_u, async>(channels_, channel_count_, global_state_);
        }

        async_rpc(const async_rpc &) = delete;

        async_rpc(async_rpc &&) noexcept = delete;

        async_rpc &operator=(async_rpc &&) noexcept = delete;

        ~async_rpc() {
            while (!keep_running_listener_); /* We wait to be sure that the thread had time to start */
            *keep_running_listener_ = false;
            listener_.join();
            sycl::free(channels_, q_);
            sycl::free(global_state_, q_);
        }

        /**
         * Returns the number of bytes that will need to be allocated on the HOST with sycl::malloc_host
         */
        [[nodiscard]] static size_t required_alloc_size(size_t channel_count) {
            return channel_count * sizeof(rpc_channel_t) + sizeof(rpc::global_state_data);
        }

        /**
         * @param q Queue to test
         * @return returns whether the queue supports the API
         */
        [[nodiscard]] static bool has_support(const sycl::queue &q) {
            return q.get_device().has(sycl::aspect::usm_host_allocations) && // Asynchronous RPC requires a device with aspect::usm_host_allocations
                   q.get_device().has(sycl::aspect::usm_atomic_host_allocations); // Asynchronous RPC requires a device with aspect::usm_atomic_host_allocations
        }


    };


}
