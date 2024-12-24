#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <omp.h>
#include <chrono>

using namespace std;

// A function for generating a random matrix
void generateMatrix(vector<vector<int>>& matrix, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            matrix[i][j] = rand() % 1000;
        }
    }
}

// A function for searching for the maximum among the minimum elements of strings
int findMaxOfMins(const vector<vector<int>>& matrix, int rows, int cols, int num_threads) {
    int max_min = INT_MIN;

#pragma omp parallel for reduction(max:max_min) num_threads(num_threads)
    for (int i = 0; i < rows; ++i) {
        int row_min = INT_MAX;
        for (int j = 0; j < cols; ++j) {
            if (matrix[i][j] < row_min) {
                row_min = matrix[i][j];
            }
        }
        max_min = max(max_min, row_min);
    }

    return max_min;
}

int main() {
    // Dimensions of the matrix (rows x cols)
    vector<pair<int, int>> matrix_sizes = { {100, 100}, {1000, 1000}, {5000, 5000}, {10000, 10000} };
    vector<int> thread_counts = { 1, 2, 4, 8 }; 
    
    cout << "Matrix Size | Threads | Execution Time (s) | Speedup | Max of Min" << endl;

    
    for (auto& size : matrix_sizes) {
        int rows = size.first;
        int cols = size.second;

        vector<vector<int>> matrix(rows, vector<int>(cols));
        generateMatrix(matrix, rows, cols);

        auto start_single = chrono::high_resolution_clock::now();
        int result_single = findMaxOfMins(matrix, rows, cols, 1);
        auto end_single = chrono::high_resolution_clock::now();
        chrono::duration<double> single_thread_time = end_single - start_single;

        cout << rows << "x" << cols << "   | " << 1
            << "       | " << single_thread_time.count()
            << "             | " << 1
            << "         | " << result_single
            << endl;

        for (int threads : thread_counts) {
            if (threads == 1) continue; 

            auto start = chrono::high_resolution_clock::now();
            int result = findMaxOfMins(matrix, rows, cols, threads);
            auto end = chrono::high_resolution_clock::now();
            chrono::duration<double> execution_time = end - start;

            double speedup = single_thread_time.count() / execution_time.count();

            cout << rows << "x" << cols << "   | " << threads
                << "       | " << execution_time.count()
                << "             | " << speedup
                << "         | " << result
                << endl;
        }
    }

    return 0;
}
