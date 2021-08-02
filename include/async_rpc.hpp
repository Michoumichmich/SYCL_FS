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

@Author Michel Migdal
*/



#pragma once

#include <sycl/sycl.hpp>
#include <thread>
#include <optional>
#include <atomic>

namespace sycl {
    namespace rpc {

        /**
         * State in which a channel can be. These states are not updated atomically
         * as it's not supported (but really it should as it's an integer)
         */
        enum class rpc_channel_state : int32_t {
            unused /** Channel free to start using it */,
            transient_state /** Data partially written */,
            ready_to_execute /** Just waiting for a runner thread to execute the request */,
            running /** Currently computing the result in sync manner */,
            async_wait [[maybe_unused]]  /** Currently computing the result in async manner */,
            result_ready /** Indicates whether the function finished running  */
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
            bool allowed_to_spawn_host_thread_ = true;
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
            bool acquire_channel() {
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
                assert(false && "How did that happen? Was the channel released without being acquired or released twice?");
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
             * @return
             */
            bool can_start_executing() {
                //
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
            functions_defined get_function() const {
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
            func_args_u get_func_args() const {
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
            ret_val_u get_retval() const {
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
     * All functions are executed her from the DEVICE
     * @tparam functions_defined
     * @tparam func_args_u
     * @tparam ret_val_u
     */
    template<typename functions_defined, typename func_args_u, typename ret_val_u, bool is_async>
    class rpc_accessor {
    private:

        /**
         * The accessor does not own the channels, so this field is mutable as it will
         * be changed by the host to establish the communication.
         */
        mutable rpc::rpc_channel<functions_defined, func_args_u, ret_val_u> *channels_;
        const size_t channel_count_;

    public:
        /**
         * Constructor not meant to be called by the user
         * @param channels
         * @param channel_count
         */
        rpc_accessor(rpc::rpc_channel<functions_defined, func_args_u, ret_val_u> *channels, size_t channel_count)
                :
                channels_(channels),
                channel_count_(channel_count) {

        }

        /**
         * If we're in synchronous mode, the call is blocking, but automatically releases the channel and returns
         * a `std::optional` with the result, on success.
         *
         * If we're in async mode, the call is non blocking, does not release the channel and using the value returned is undefined.
         * The optional still indicated whether the call succeeded.
         *
         * @tparam func Function to call
         * @tparam do_acquire_channel We can ask not to acquire the channel if we pre-acquired it
         * @tparam do_release_channel ONLY SYNC: We can ask not to release the channel if we want to use it after.
         * @param channel_idx Index of the channel used
         * @param args function arguments squashed in an union (but not necessarily)
         * @param can_spawn_host_thread indicates whether we allow the RPC to be launched in a separate thread on the side of the host
         * this can be useful to be set to false if we know that the function call is quick. But if we're doing ASYNC RPC and on the host, the function call is
         * blocking, it's not great. At the same time, launching inexpensive small functions in a thread can be very costly too.
         * @return the return an optional indicating whether the call succeeded.
         * The contained value is the result only if we're in asynchronous mode,
         * else it's undefined and we need to get the result through a synchronisation function.
         */
        template<functions_defined func, bool do_acquire_channel = true, bool do_release_channel = true>
        std::optional<ret_val_u> call_remote_procedure(size_t channel_idx, const func_args_u &args, bool can_spawn_host_thread = true) const {
            if constexpr(do_acquire_channel) {
                if (channel_idx >= channel_count_ || !channels_[channel_idx].acquire_channel()) {
                    return std::nullopt; // Channel not acquired
                }
            }

            // Channel is in transient state
            channels_[channel_idx].set_function(func);
            channels_[channel_idx].set_func_args(args);
            channels_[channel_idx].set_allowed_to_spawn_host_thread(can_spawn_host_thread);
            channels_[channel_idx].set_executable();

            if constexpr(!is_async) {
                wait(channel_idx);
                auto result = channels_[channel_idx].get_retval();
                if constexpr(do_release_channel) {
                    release(channel_idx);
                }
                return result;
            }

            // We cannot return a std::nullopt as the call must have succeeded, but
            return channels_[channel_idx].get_retval();
        }

        /**
         * ASYNC ONLY
         * Wait's for the result.
         * @param channel_idx
         */
        void wait(size_t channel_idx) const {
            while (!channels_[channel_idx].is_result_ready()) {}
        }

        void clear(size_t channel_idx) const {
            channels_[channel_idx].set_retval({});
            channels_[channel_idx].set_func_args({});
        }

        /**
         * ASYNC ONLY & BLOCKING
         * Returns an asynchronous result. Calling if with a sync api results in UB.
         * @param channel_idx
         * @return
         */
        ret_val_u get_result(size_t channel_idx) const {
            wait(channel_idx);
            return channels_[channel_idx].get_retval();
        }

        /**
         * ASYNC ONLY & NON-BLOCKING
         * Returns an asynchronous result if available. Calling if with a sync api results in UB.
         * @param channel_idx
         * @return
         */
        std::optional<ret_val_u> try_get_result(size_t channel_idx) const {
            if (channels_[channel_idx].is_result_ready()) {
                return channels_[channel_idx].get_retval();
            }
            return std::nullopt;
        }

        /**
         * ASYNC ONLY release the channel after getting your result back.
         * This increase the lifetime of the communication channel which allows
         * to perform other operations between the host and the device such as memcpy between buffers
         * that are tied to a channel_idx.
         * @param channel_idx
         */
        void release(size_t channel_idx) const {
            wait(channel_idx);
            clear(channel_idx);
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
        bool acquire(size_t channel_idx) const {
            return channels_[channel_idx].acquire_channel();
        }
    };

    template<typename functions_defined, typename func_args_u, typename ret_val_u, bool parallel_runners>
    class async_rpc {
    private:
        using rpc_channel_t = rpc::rpc_channel<functions_defined, func_args_u, ret_val_u>;
        rpc::rpc_channel<functions_defined, func_args_u, ret_val_u> *channels_ = nullptr;
        sycl::queue q_;
        size_t channel_count_;
        volatile bool *keep_running_listener_ = nullptr; //DANGEROUS we're probably doing bad things by returning the reference of a local variable, but we can control the lifetime of the function so...
        std::thread listener_;

        static inline void thread_runner(void (*f)(rpc_channel_t *), rpc_channel_t *channels, size_t count, volatile bool **class_switch_address, double frequency) {
            volatile bool keep_running_listener = true;
            const std::chrono::nanoseconds sleep_time = std::chrono::nanoseconds((frequency > 0) ? (int64_t) ((1e9 / frequency)) : 0);
            *class_switch_address = &keep_running_listener;
            std::vector<std::atomic<int64_t>> running_channels(count);
            for (auto &item : running_channels) {
                item = false;
            }
            std::vector<std::thread> running_threads(count);
            while (keep_running_listener) {
                for (size_t i = 0; i < count && keep_running_listener; ++i) {
                    int64_t expected_false_running_state = false;
                    // We check if the channel can be executed, if so then we acquire a lock
                    if (channels[i].can_start_executing() && running_channels[i].compare_exchange_strong(expected_false_running_state, true)) {
                        //We clear the thread
                        auto func = [=, &running_channels]() {
                            channels[i].set_as_executing();
                            f(channels + i);
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

        }

    public:

        async_rpc(size_t channel_count, const sycl::queue &q, void (*rpc_channel_runner)(rpc_channel_t *), double runner_frequency = -1)
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

            listener_ = std::thread(thread_runner, rpc_channel_runner, channels_, channel_count_, &keep_running_listener_, runner_frequency);
        }

        template<bool async>
        rpc_accessor<functions_defined, func_args_u, ret_val_u, async> get_access() const {
            return rpc_accessor<functions_defined, func_args_u, ret_val_u, async>(channels_, channel_count_);
        }

        async_rpc(const async_rpc &) = delete;

        async_rpc(async_rpc &&) noexcept = default;

        async_rpc &operator=(async_rpc &&) noexcept = default;

        ~async_rpc() {
            while (!keep_running_listener_); /* We wait to be sure that the thread had time to start */
            *keep_running_listener_ = false;
            listener_.join();
            sycl::free(channels_, q_);
        }

        static size_t required_alloc_size(size_t channel_count) {
            return channel_count * sizeof(rpc_channel_t);
        }

        static bool has_support(const sycl::queue &q) {
            return q.get_device().has(sycl::aspect::usm_host_allocations) && // Asynchronous RPC requires a device with aspect::usm_host_allocations
                   q.get_device().has(sycl::aspect::usm_atomic_host_allocations); // Asynchronous RPC requires a device with aspect::usm_atomic_host_allocations
        }


    };


}
