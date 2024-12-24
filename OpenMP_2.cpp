#include <iostream>
#include <vector>
#include <omp.h>
#include <chrono>

using namespace std;

// Function for calculating the scalar product
double scalar_product(const vector<double>& vec1, const vector<double>& vec2, int num_threads) {
    double result = 0.0;

#pragma omp parallel for reduction(+:result) num_threads(num_threads)
    for (size_t i = 0; i < vec1.size(); ++i) {
        result += vec1[i] * vec2[i];
    }

    return result;
}

int main() {
    vector<int> vector_sizes = { 1000, 10000, 100000, 1000000 }; 
    vector<int> thread_counts = { 1, 2, 4, 8 };

    cout << "Vector Size | Threads | Reduction Time (s) | Speedup (Reduction) | Scalar Product" << endl;

    for (int size : vector_sizes) {
        vector<double> vec1(size, 1.0);  // vector a
        vector<double> vec2(size, 2.0);  // vector b

        // 1 thread
        auto start_single = chrono::high_resolution_clock::now();
        double result_single = scalar_product(vec1, vec2, 1);
        auto end_single = chrono::high_resolution_clock::now();
        chrono::duration<double> single_thread_time = end_single - start_single;

        cout << size << "         | " << 1
            << "       | " << single_thread_time.count()
            << "             | " << 1
            << "              | " << result_single << endl;

        // for other threads
        for (int threads : thread_counts) {
            if (threads == 1) continue;

            auto start = chrono::high_resolution_clock::now();
            double result = scalar_product(vec1, vec2, threads);
            auto end = chrono::high_resolution_clock::now();
            chrono::duration<double> reduction_time = end - start;

            // speedup
            double speedup = single_thread_time.count() / reduction_time.count();

            // results
            cout << size << "         | " << threads
                << "       | " << reduction_time.count()
                << "             | " << speedup
                << "              | " << result << endl; 
        }
    }

    return 0;
}
