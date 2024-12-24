#include <iostream>
#include <vector>
#include <omp.h>
#include <limits>
#include <cstdlib>
#include <ctime>

int main() {
    const std::vector<int> vector_sizes = { 1000, 10000, 100000, 1000000 }; // Размерности векторов
    const std::vector<int> thread_counts = { 1, 2, 4, 8 };                  // Количество потоков

    std::cout << "Vector Size | Threads | Reduction Time (s) | No Reduction Time (s) | Speedup (Reduction) | Speedup (No Reduction)\n";

    for (int size : vector_sizes) {
        std::vector<int> data(size);

        // Заполняем вектор случайными числами
        std::srand(std::time(nullptr));
        for (int i = 0; i < size; i++) {
            data[i] = std::rand() % 1000; // Диапазон значений от 0 до 999
        }

        // Переменные для времени с одним потоком (для редукции)
        omp_set_num_threads(1);

        int min_value = std::numeric_limits<int>::max();
        int max_value = std::numeric_limits<int>::min();

        // Измеряем время выполнения с одним потоком с редукцией
        double start_time = omp_get_wtime();
        #pragma omp parallel for reduction(min:min_value) reduction(max:max_value)
        for (int i = 0; i < size; i++) {
            if (data[i] < min_value) min_value = data[i];
            if (data[i] > max_value) max_value = data[i];
        }
        double one_thread_reduction_time = omp_get_wtime() - start_time;

        min_value = std::numeric_limits<int>::max();
        max_value = std::numeric_limits<int>::min();

        // Измеряем время выполнения с одним потоком без редукции
        start_time = omp_get_wtime();
        #pragma omp parallel
        {
            int local_min = std::numeric_limits<int>::max();
            int local_max = std::numeric_limits<int>::min();

            #pragma omp for
            for (int i = 0; i < size; i++) {
                if (data[i] < local_min) local_min = data[i];
                if (data[i] > local_max) local_max = data[i];
            }

            #pragma omp critical
            {
                if (local_min < min_value) min_value = local_min;
                if (local_max > max_value) max_value = local_max;
            }
        }
        double one_thread_no_reduction_time = omp_get_wtime() - start_time;

        // Теперь выводим результаты для 1 потока
        std::cout << size << "         | "
            << 1 << "       | "
            << one_thread_reduction_time << "             | "
            << one_thread_no_reduction_time << "               | "
            << 1.0 << "              | "
            << 1.0 << "\n"; // Для 1 потока ускорение всегда 1

        // Для остальных потоков
        for (int threads : thread_counts) {
            if (threads == 1) continue; // Пропускаем уже измеренные для 1 потока

            omp_set_num_threads(threads);

            min_value = std::numeric_limits<int>::max();
            max_value = std::numeric_limits<int>::min();

            // Измеряем время выполнения с редукцией
            start_time = omp_get_wtime();
            #pragma omp parallel for reduction(min:min_value) reduction(max:max_value)
            for (int i = 0; i < size; i++) {
                if (data[i] < min_value) min_value = data[i];
                if (data[i] > max_value) max_value = data[i];
            }
            double reduction_time = omp_get_wtime() - start_time;

            // Без редукции
            min_value = std::numeric_limits<int>::max();
            max_value = std::numeric_limits<int>::min();

            // Измеряем время выполнения без редукции
            start_time = omp_get_wtime();
            #pragma omp parallel
            {
                int local_min = std::numeric_limits<int>::max();
                int local_max = std::numeric_limits<int>::min();

                #pragma omp for
                for (int i = 0; i < size; i++) {
                    if (data[i] < local_min) local_min = data[i];
                    if (data[i] > local_max) local_max = data[i];
                }

                #pragma omp critical
                {
                    if (local_min < min_value) min_value = local_min;
                    if (local_max > max_value) max_value = local_max;
                }
            }
            double no_reduction_time = omp_get_wtime() - start_time;

            // Рассчитываем ускорение для редукции и без редукции (t на одном/t текущее)
            double speedup_reduction = one_thread_reduction_time / reduction_time;
            double speedup_no_reduction = one_thread_no_reduction_time / no_reduction_time;

            // Вывод результатов
            std::cout << size << "         | "
                << threads << "       | "
                << reduction_time << "             | "
                << no_reduction_time << "               | "
                << speedup_reduction << "              | "
                << speedup_no_reduction << "\n";
        }
    }

    return 0;
}
