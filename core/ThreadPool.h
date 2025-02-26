#ifndef THREAD_POOL_H
#define THREAD_POOL_H

#include <vector>
#include <queue>
#include <memory>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <future>
#include <functional>
#include <stdexcept>

#include <iostream>
#include <bits/stdc++.h>

class ThreadPool {
public:
    ThreadPool(size_t);
    template<class F, class... Args>
    auto enqueue(F&& f, Args&&... args) 
        -> std::future<typename std::result_of<F(Args...)>::type>;
    ~ThreadPool();
private:
    // need to keep track of threads so we can join them
    std::vector< std::thread > workers;
    // the task queue
    std::queue< std::function<void()> > tasks;
    
    // synchronization
    std::mutex queue_mutex;
    std::condition_variable condition;
    bool stop;
};
 
// the constructor just launches some amount of workers
inline ThreadPool::ThreadPool(size_t threads)
    :   stop(false)
{
    for(size_t i = 0;i<threads;++i)
        workers.emplace_back(
            [this]
            {
                for(;;)
                {
                    std::function<void()> task;

                    std::thread::id id = std::this_thread::get_id();
                    // std::cout << std::hash<std::thread::id>()(id) << std::endl;
                    {
                        #ifdef ANNIWO_INTERNAL_DEBUG
                        std::cout<<"ThreadPool worker calling lock "<<this->stop<<this->tasks.size()<<",th:"<<std::hash<std::thread::id>()(id)<<std::endl;
                        #endif

                        std::unique_lock<std::mutex> lock(this->queue_mutex);
                        
                        #ifdef ANNIWO_INTERNAL_DEBUG
                        std::cout<<"ThreadPool worker called lock "<<this->stop<<this->tasks.size()<<",th:"<<std::hash<std::thread::id>()(id)<<std::endl;
                        #endif

                        if(this->stop )
                        {
                            #ifdef ANNIWO_INTERNAL_DEBUG
                            std::cout<<"ThreadPool worker returned "<<this->stop<<this->tasks.size()<<",th:"<<std::hash<std::thread::id>()(id)<<std::endl;
                            #endif

                            return;
                        }
                        //xiangbin:也就是说，一旦带有pred的wait被notify的时候，它会去检查谓词对象的bool返回值是否是true, 如果是true才真正唤醒，否则继续block
                        //当销毁threadpool时候此处可能造成死锁，因为当tasks.empty为true时候
                        #ifdef ANNIWO_INTERNAL_DEBUG
                        std::cout<<"ThreadPool worker calling condition.wait "<<this->stop<<this->tasks.size()<<",th:"<<std::hash<std::thread::id>()(id)<<std::endl;
                        #endif

                        this->condition.wait(lock,
                            [this]{ return this->stop || !this->tasks.empty(); });
                        
                        #ifdef ANNIWO_INTERNAL_DEBUG
                        std::cout<<"ThreadPool worker called condition.wait "<<this->stop<<this->tasks.size()<<",th:"<<std::hash<std::thread::id>()(id)<<std::endl;
                        #endif

                        if(this->stop )
                        {
                            #ifdef ANNIWO_INTERNAL_DEBUG
                            std::cout<<"ThreadPool worker returned "<<this->stop<<this->tasks.size()<<",th:"<<std::hash<std::thread::id>()(id)<<std::endl;
                            #endif

                            return;
                        }
                        task = std::move(this->tasks.front());
                        this->tasks.pop();
                    }

                    #ifdef ANNIWO_INTERNAL_DEBUG
                    std::cout<<"ThreadPool worker calling task "<<this->stop<<this->tasks.size()<<",th:"<<std::hash<std::thread::id>()(id)<<std::endl;
                    #endif

                    task();

                    #ifdef ANNIWO_INTERNAL_DEBUG
                    std::cout<<"ThreadPool worker called task "<<this->stop<<this->tasks.size()<<",th:"<<std::hash<std::thread::id>()(id)<<std::endl;
                    #endif


                }
            }
        );
}

// add new work item to the pool
template<class F, class... Args>
auto ThreadPool::enqueue(F&& f, Args&&... args) 
    -> std::future<typename std::result_of<F(Args...)>::type>
{
    using return_type = typename std::result_of<F(Args...)>::type;

    auto task = std::make_shared< std::packaged_task<return_type()> >(
            std::bind(std::forward<F>(f), std::forward<Args>(args)...)
        );

    // auto task = new std::packaged_task<return_type()>(
    //         std::bind(std::forward<F>(f), std::forward<Args>(args)...)
    //     );
        
    std::future<return_type> res = task->get_future();
    {
        std::unique_lock<std::mutex> lock(queue_mutex);

        // don't allow enqueueing after stopping the pool
        if(stop)
            throw std::runtime_error("enqueue on stopped ThreadPool");

        // tasks.emplace([task](){ (*task)(); delete task; });
        tasks.emplace([task](){ (*task)(); });
    }
    condition.notify_one();
    return res;
}


// the destructor joins all threads
inline ThreadPool::~ThreadPool()
{
    {
        std::unique_lock<std::mutex> lock(queue_mutex);
        stop = true;
#ifdef ANNIWO_INTERNAL_DEBUG
        std::cout<<"ThreadPool entered ~ this->tasks.size:"<<this->tasks.size() <<"workers.size"<<workers.size()<<std::endl;
#endif
    }
    condition.notify_all();

#ifdef ANNIWO_INTERNAL_DEBUG
    std::cout<<"ThreadPool ~ this->tasks.size:"<<this->tasks.size() <<"workers.size"<<workers.size()<<std::endl;
#endif

    for(std::thread &worker: workers)
    {
// #ifdef ANNIWO_INTERNAL_DEBUG
        std::cout<<"ThreadPool ~ joining th:"<<std::hash<std::thread::id>()(worker.get_id())<<" worker this->tasks.size:"<<this->tasks.size() <<"workers.size"<<workers.size()<<std::endl;
// #endif

        worker.join();
// #ifdef ANNIWO_INTERNAL_DEBUG
        std::cout<<"ThreadPool ~ joined  th:"<<std::hash<std::thread::id>()(worker.get_id())<<"  worker this->tasks.size:"<<this->tasks.size() <<"workers.size"<<workers.size()<<std::endl;
// #endif


    }
}

#endif
