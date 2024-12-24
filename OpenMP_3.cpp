#include <iostream>
#include <vector>
#include <cmath>
#include <omp.h>
#include <chrono>

using namespace std;

// A function for calculating the value of the function f(x) = x^2
double f(double x) {
    return x * x;  
}

// A function for calculating a certain integral using the rectangle method
double integral(double a, double b, int N, int num_threads) {
    double sum = 0.0;
    double dx = (b - a) / N;

    #pragma omp parallel for reduction(+:sum) num_threads(num_threads)
    for (int i = 0; i < N; ++i) {
        double x = a + i * dx;
        sum += f(x) * dx;
    }

    return sum;
}

int main() {
    double a = 0.0;  // The beginning of the interval
    double b = 1.0;  // The end of the interval
    vector<int> vector_sizes = { 1000, 10000, 100000, 1000000 };  // Number of partitions
    vector<int> thread_counts = { 1, 2, 4, 8 };

    cout << "Vector Size | Threads | Integration Time (s) | Speedup (Integration) | Integration Product" << endl;

    for (int N : vector_sizes) {
        // for 1 thread
        auto start_single = chrono::high_resolution_clock::now();
        double result_single = integral(a, b, N, 1);
        auto end_single = chrono::high_resolution_clock::now();
        chrono::duration<double> single_thread_time = end_single - start_single;

        cout << N << "         | " << 1
            << "       | " << single_thread_time.count()
            << "             | " << 1
            << "             | " << result_single
            << endl;

        for (int threads : thread_counts) {
            if (threads == 1) continue;

            auto start = chrono::high_resolution_clock::now();
            double result = integral(a, b, N, threads);
            auto end = chrono::high_resolution_clock::now();
            chrono::duration<double> integration_time = end - start;

            double speedup = single_thread_time.count() / integration_time.count();

            cout << N << "         | " << threads
                << "       | " << integration_time.count()
                << "             | " << speedup
                << "             | " << result
                << endl;
        }
    }

    return 0;
}
