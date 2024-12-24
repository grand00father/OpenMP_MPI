#include <mpi.h>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <numeric>

void initializeMatrix(std::vector<double>& matrix, int rows, int cols) {
    for (int i = 0; i < rows * cols; ++i) {
        matrix[i] = rand() % 10; // Random values between 0 and 9
    }
}

void printMatrix(const std::vector<double>& matrix, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            std::cout << matrix[i * cols + j] << " ";
        }
        std::cout << std::endl;
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    std::vector<int> matrix_sizes = { 16, 32, 64, 128, 256, 512, 1024, 2048 }; 
    std::vector<int> process_counts = { 1, 2, 4, 8, 16, 32, 64 };

   
    for (int processes : process_counts) {
        if (rank == 0) {
            std::cout << "Running with " << processes << " processes..." << std::endl;
        }

       
        for (int N : matrix_sizes) {
            if (rank == 0) {
                std::cout << "Matrix size: " << N << " x " << N << ", Number of processes: " << processes << std::endl;
            }

            
            int rowsPerProcess = N / processes;
            int remainder = N % processes;

            
            std::vector<int> rowsPerProc(processes, rowsPerProcess);
            for (int i = 0; i < remainder; ++i) {
                rowsPerProc[i] += 1;
            }

            
            std::vector<double> A, B(N * N), C(rowsPerProc[rank] * N, 0);
            std::vector<double> localA(rowsPerProc[rank] * N);

            if (rank == 0) {
                A.resize(N * N);
                initializeMatrix(A, N, N);
                initializeMatrix(B, N, N);
            }

            
            if (rank == 0) {
                for (int i = 1; i < processes; ++i) {
                    MPI_Send(A.data() + std::accumulate(rowsPerProc.begin(), rowsPerProc.begin() + i, 0) * N,
                        rowsPerProc[i] * N, MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
                }
                std::copy(A.data(), A.data() + rowsPerProc[0] * N, localA.data());
            }
            else {
                MPI_Recv(localA.data(), rowsPerProc[rank] * N, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }

            
            MPI_Bcast(B.data(), N * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

            
            double startTime = MPI_Wtime();

           
            for (int i = 0; i < rowsPerProc[rank]; ++i) {
                for (int j = 0; j < N; ++j) {
                    for (int k = 0; k < N; ++k) {
                        C[i * N + j] += localA[i * N + k] * B[k * N + j];
                    }
                }
            }

            double endTime = MPI_Wtime();
            double localTime = endTime - startTime;

            
            double maxTime;
            MPI_Reduce(&localTime, &maxTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

            
            if (rank == 0) {
                std::vector<double> fullC(N * N, 0);
                std::copy(C.begin(), C.end(), fullC.begin());

                for (int i = 1; i < processes; ++i) {
                    MPI_Recv(fullC.data() + std::accumulate(rowsPerProc.begin(), rowsPerProc.begin() + i, 0) * N,
                        rowsPerProc[i] * N, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                }

                std::cout << "Parallel execution time: " << maxTime << " seconds\n";

                
                double serialTime = maxTime * processes; 
                double speedup = (serialTime > 0) ? serialTime / maxTime : 1.0;
                std::cout << "Speedup for " << processes << " processes: " << speedup << std::endl;
            }
            else {
                MPI_Send(C.data(), rowsPerProc[rank] * N, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
            }

            MPI_Barrier(MPI_COMM_WORLD); 
        }
    }

    MPI_Finalize();
    return 0;
}
