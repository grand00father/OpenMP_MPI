#include <mpi.h>
#include <iostream>
#include <vector>
#include <numeric> 
#include <cstdlib> 
#include <ctime> 
#include <cmath> 

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    std::vector<int> vector_sizes = { 2000, 4000, 6000, 8000, 10000 };
    std::vector<int> process_counts = { 1, 2, 3, 4, 5, 6, 7, 8 };

    for (int vector_size : vector_sizes) {
        double time_single_process = 0; 

        if (rank == 0) {
            std::cout << "Vector size: " << vector_size << std::endl;
        }

        std::vector<int> vec_a, vec_b;
        if (rank == 0) {
            vec_a.resize(vector_size);
            vec_b.resize(vector_size);


            std::srand(std::time(nullptr));
            for (int i = 0; i < vector_size; ++i) {
                vec_a[i] = std::rand() % 10; 
                vec_b[i] = std::rand() % 10;
            }
        }

     
        for (int processes : process_counts) {
            if (rank == 0) {
                std::cout << "Running with " << processes << " processes..." << std::endl;
            }

            int local_size = std::ceil(static_cast<double>(vector_size) / processes);
            std::vector<int> local_a(local_size), local_b(local_size);

            MPI_Scatter(vec_a.data(), local_size, MPI_INT, local_a.data(), local_size, MPI_INT, 0, MPI_COMM_WORLD);
            MPI_Scatter(vec_b.data(), local_size, MPI_INT, local_b.data(), local_size, MPI_INT, 0, MPI_COMM_WORLD);

            double start_time = MPI_Wtime();

            int local_dot_product = std::inner_product(local_a.begin(), local_a.end(), local_b.begin(), 0);

            int global_dot_product = 0;
            MPI_Reduce(&local_dot_product, &global_dot_product, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

            double end_time = MPI_Wtime();

            double time_taken = end_time - start_time;

            if (processes == 1) {
                time_single_process = time_taken;
            }

            if (rank == 0) {
                std::cout << "Time taken for " << vector_size << " elements with " << processes << " processes: "
                    << time_taken << " seconds" << std::endl;

                double speedup = (time_single_process > 0) ? time_single_process / time_taken : 1.0;
                std::cout << "Speedup for " << processes << " processes: " << speedup << std::endl;
            }

            // ќжидаем завершени€ всех процессов
            MPI_Barrier(MPI_COMM_WORLD);
        }

        std::cout << "-------------------------------------" << std::endl;
    }

    MPI_Finalize();
    return 0;
}
