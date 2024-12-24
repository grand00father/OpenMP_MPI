#include <mpi.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <ctime>
#include <chrono>

void generate_random_vector(std::vector<int>& vec, int size) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(1, 100);

    vec.resize(size);
    for (int i = 0; i < size; ++i) {
        vec[i] = dis(gen);
    }
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int max_vector_size = 10000; //Maximum vector size
    const int step = 2000; // A step to increase the size of the vector

    double time_one_process = 0.0;

    // We vary the number of processes from 1 to size
    for (int num_processes = 1; num_processes <= size; ++num_processes) {

        for (int vector_size = step; vector_size <= max_vector_size; vector_size += step) {
            
            auto start_time = std::chrono::high_resolution_clock::now();

            std::vector<int> data;

            if (rank == 0) {
                generate_random_vector(data, vector_size);
            }

            // Block size for each process
            int local_size = vector_size / num_processes;
            if (rank == num_processes - 1) {
                local_size += vector_size % num_processes;
            }

            std::vector<int> local_data(local_size);

            // Distribute the vector among the processes
            MPI_Scatter(data.data(), local_size, MPI_INT, local_data.data(), local_size, MPI_INT, 0, MPI_COMM_WORLD);

            // local min and max
            int local_min = *std::min_element(local_data.begin(), local_data.end());
            int local_max = *std::max_element(local_data.begin(), local_data.end());

            // Collecting global minimum and maximum
            int global_min, global_max;
            MPI_Reduce(&local_min, &global_min, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);
            MPI_Reduce(&local_max, &global_max, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);

            auto end_time = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> duration = end_time - start_time;

            if (rank == 0) {
                if (num_processes == 1) {
                    time_one_process = duration.count();
                }

                std::cout << "Vector size: " << vector_size << ", Processes: " << num_processes << std::endl;
                std::cout << "Global minimum: " << global_min << std::endl;
                std::cout << "Global maximum: " << global_max << std::endl;
                std::cout << "Time taken: " << duration.count() << " seconds" << std::endl;

                if (time_one_process > 0.0) {
                    double speedup = time_one_process / duration.count();
                    std::cout << "Speedup: " << speedup << std::endl;
                }

                std::cout << "---------------------------------------------" << std::endl;
            }

            MPI_Barrier(MPI_COMM_WORLD); // Синхронизация всех процессов перед следующей итерацией
        }
    }

    MPI_Finalize();
    return 0;
}
