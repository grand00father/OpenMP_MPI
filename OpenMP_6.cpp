#include <iostream>
#include <vector>
#include <omp.h>
#include <random>
#include <chrono>

// A function for heavy calculations
int heavy_computation(int value) {
    int result = value;
    for (int i = 0; i < 10000; ++i) { 
        result += rand() % 1000;
    }
    return result;
}

// A function for easy calculations
int light_computation(int value) {
    return value + 1; 
}

// Diffirent distribution
void run_experiment(const std::string& dist, int num_iterations, int num_threads) {
    std::vector<int> data(num_iterations);

    std::default_random_engine generator;
    std::uniform_int_distribution<int> distribution_range(0, 1000);
    for (int i = 0; i < num_iterations; ++i) {
        data[i] = distribution_range(generator);
    }

    omp_set_num_threads(num_threads);

    if (dist == "static") {
        omp_set_schedule(omp_sched_static, 0);
    }
    else if (dist == "dynamic") {
        omp_set_schedule(omp_sched_dynamic, 0);
    }
    else if (dist == "guided") {
        omp_set_schedule(omp_sched_guided, 0);
    }

    auto start = std::chrono::high_resolution_clock::now();

    #pragma omp parallel for
    for (int i = 0; i < num_iterations; ++i) {
        if (i % 10 == 0) {
            // Every tenth iteration is a heavy operation.
            data[i] = heavy_computation(data[i]);
        }
        else {
            // Easy operation on the remaining iterations
            data[i] = light_computation(data[i]);
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;

    std::cout << num_iterations << ", " << num_threads << ", " << duration.count() << ", " << dist << std::endl;
}

int main() {
    std::cout << "Iterations | Threads | Execution Time (s) | Distribution" << std::endl;

    std::vector<int> iterations = { 100, 1000, 10000 };

    std::vector<std::string> distributions = { "static", "dynamic", "guided" };

    for (int num_threads = 1; num_threads <= 8; num_threads *= 2) {
        for (int num_iterations : iterations) {
            for (const std::string& dist : distributions) {
                run_experiment(dist, num_iterations, num_threads);
            }
        }
    }

    return 0;
}
