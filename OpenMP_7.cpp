#include <iostream>
#include <vector>
#include <omp.h>
#include <chrono>
#include <mutex>

std::mutex mtx;

// A function for summing array elements using atomic operations
double sum_atomic(const std::vector<int>& data) {
    double sum = 0;
    #pragma omp parallel for 
    for (size_t i = 0; i < data.size(); ++i) {
        #pragma omp atomic
        sum += data[i];
    }
    return sum;
}

// A function for summing array elements using a critical section
double sum_critical(const std::vector<int>& data) {
    double sum = 0;
    #pragma omp parallel for
    for (size_t i = 0; i < data.size(); ++i) {
        #pragma omp critical
        {
            sum += data[i];
        }
    }
    return sum;
}

// A function for summing array elements using locks
double sum_lock(const std::vector<int>& data) {
    double sum = 0;
    #pragma omp parallel for
    for (size_t i = 0; i < data.size(); ++i) {
        mtx.lock();
        sum += data[i];
        mtx.unlock();
    }
    return sum;
}

// A function for summing array elements using standard OpenMP reduction
double sum_reduction(const std::vector<int>& data) {
    double sum = 0;
    #pragma omp parallel for reduction(+:sum)
    for (size_t i = 0; i < data.size(); ++i) {
        sum += data[i];
    }
    return sum;
}

void run_experiment(const std::vector<int>& data, int num_threads) {
    omp_set_num_threads(num_threads);

    auto start = std::chrono::high_resolution_clock::now();
    double result_atomic = sum_atomic(data);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_atomic = end - start;
    std::cout << data.size() << ", " << num_threads << ", Atomic, " << duration_atomic.count() << "s" << std::endl;

    start = std::chrono::high_resolution_clock::now();
    double result_critical = sum_critical(data);
    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_critical = end - start;
    std::cout << data.size() << ", " << num_threads << ", Critical, " << duration_critical.count() << "s" << std::endl;

    start = std::chrono::high_resolution_clock::now();
    double result_lock = sum_lock(data);
    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_lock = end - start;
    std::cout << data.size() << ", " << num_threads << ", Lock, " << duration_lock.count() << "s" << std::endl;

    start = std::chrono::high_resolution_clock::now();
    double result_reduction = sum_reduction(data);
    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_reduction = end - start;
    std::cout << data.size() << ", " << num_threads << ", OpenMP reduction, " << duration_reduction.count() << "s" << std::endl;
}

int main() {
    std::vector<size_t> sizes = { 100000, 1000000, 10000000, 50000000 }; 

    for (size_t size : sizes) {
        std::vector<int> data(size);
        for (size_t i = 0; i < size; ++i) {
            data[i] = rand() % 100;
        }

        for (int num_threads = 1; num_threads <= 8; num_threads *= 2) {
            run_experiment(data, num_threads);
        }
    }

    return 0;
}
