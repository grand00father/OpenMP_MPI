#include <iostream>
#include <omp.h>
#include <vector>
#include <random>
#include <algorithm>
#include <chrono>
#include <climits>

using namespace std;

// A function for generating a ribbon matrix
vector<vector<int>> generate_band_matrix(int n, int k) {
    vector<vector<int>> matrix(n, vector<int>(n, 0));

    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> dis(1, 100); // Generating random numbers from 1 to 100

    // Filling in the feed
    for (int i = 0; i < n; ++i) {
        for (int j = max(0, i - k); j <= min(n - 1, i + k); ++j) {
            matrix[i][j] = dis(gen);
        }
    }
    return matrix;
}

int max_of_min_elements(const vector<vector<int>>& matrix, int threads, const string& distribution) {
    int n = matrix.size();
    int max_of_mins = INT_MIN; 

    omp_set_num_threads(threads);


    if (distribution == "static") {
        #pragma omp parallel for schedule(static) reduction(max:max_of_mins)
        for (int i = 0; i < n; ++i) {
            int row_min = *min_element(matrix[i].begin(), matrix[i].end());
            max_of_mins = max(max_of_mins, row_min);
        }
    }
    else if (distribution == "dynamic") {
        #pragma omp parallel for schedule(dynamic) reduction(max:max_of_mins)
        for (int i = 0; i < n; ++i) {
            int row_min = *min_element(matrix[i].begin(), matrix[i].end());
            max_of_mins = max(max_of_mins, row_min); 
        }
    }
    else if (distribution == "guided") {
        #pragma omp parallel for schedule(guided) reduction(max:max_of_mins)
        for (int i = 0; i < n; ++i) {
            int row_min = *min_element(matrix[i].begin(), matrix[i].end()); 
            max_of_mins = max(max_of_mins, row_min); 
        }
    }

    return max_of_mins;
}

int main() {
    vector<int> matrix_sizes = { 100, 1000, 5000, 10000 };
    int k = 10;    // Width of the tape

    cout << "Matrix Size | Threads | Execution Time (s) | Distribution" << endl;

    for (int n : matrix_sizes) {
        vector<vector<int>> matrix = generate_band_matrix(n, k);

        for (int threads = 1; threads <= 8; threads *= 2) {
            for (const string& distribution : { "static", "dynamic", "guided" }) {
                auto start = chrono::high_resolution_clock::now();
                int result = max_of_min_elements(matrix, threads, distribution);
                auto end = chrono::high_resolution_clock::now();

                chrono::duration<double> duration = end - start;
                cout << n << "x" << n << " | "
                    << threads << " | "
                    << duration.count() << " | "
                    << distribution << endl;
            }
        }
    }

    return 0;
}
