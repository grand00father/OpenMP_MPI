#include <mpi.h>
#include <iostream>
#include <vector>
#include <chrono>

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size < 2) {
        if (rank == 0) {
            std::cerr << "Error: This program requires at least two processes.\n";
        }
        MPI_Finalize();
        return 1;
    }

    std::vector<int> message_sizes = { 1024, 2048, 4096, 8192, 16384 }; // Message sizes in bytes
    std::vector<int> iterations_list = { 100, 500, 1000, 2000 }; // Number of iterations

    std::vector<char> send_buffer, recv_buffer;

    for (int n : message_sizes) {
        for (int iterations : iterations_list) {

            //Initialize buffers for the current message size
            send_buffer.resize(n, 'x');
            recv_buffer.resize(n);

            MPI_Barrier(MPI_COMM_WORLD);

            auto start_time = std::chrono::high_resolution_clock::now();

            for (int i = 0; i < iterations; ++i) {
                if (rank == 0) {
                    // Process 0 sends a message to process 1 and receives a response
                    MPI_Send(send_buffer.data(), n, MPI_CHAR, 1, 0, MPI_COMM_WORLD);
                    MPI_Recv(recv_buffer.data(), n, MPI_CHAR, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                }
                else if (rank == 1) {
                    // Process 1 receives a message from process 0 and sends a reply
                    MPI_Recv(recv_buffer.data(), n, MPI_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    MPI_Send(send_buffer.data(), n, MPI_CHAR, 0, 0, MPI_COMM_WORLD);
                }
            }

            MPI_Barrier(MPI_COMM_WORLD);

            auto end_time = std::chrono::high_resolution_clock::now();
            double elapsed_time = std::chrono::duration<double>(end_time - start_time).count();

            if (rank == 0) {
                std::cout << "Message size: " << n << " bytes\n";
                std::cout << "Iterations: " << iterations << "\n";
                std::cout << "Total time: " << elapsed_time << " seconds\n";
                std::cout << "Average time per message pair: " << (elapsed_time / iterations) << " seconds\n";

                double bandwidth = (n * iterations * 2) / elapsed_time / 1e6; 
                std::cout << "Data transfer speed: " << bandwidth << " MB/s\n";

                double speed = static_cast<double>(n) / elapsed_time; 
                std::cout << "Speed (bytes per second): " << speed << " bytes/sec\n";

                std::cout << "-------------------------------------\n";
            }


            MPI_Barrier(MPI_COMM_WORLD);
        }
    }

    MPI_Finalize();
    return 0;
}
