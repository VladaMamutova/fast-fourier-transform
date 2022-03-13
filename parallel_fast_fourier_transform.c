#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include <math.h>
#include <complex.h>
#include <time.h>

#define PI 3.141592653589
#define N 16384 // = 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384
#define DIRECT 1 // прямое преобразование
#define INVERSE -1 // обратное преобразование
#define COLUMNS 4 // [0] - действительная часть, [1] - мнимая часть, [2] - действительная часть результата, [3] - мнимая часть результата
#define NUMBER_IS_2_POW_K(x)  (x && !(x & (x - 1)))
#define ROOT_RANK 0 // ранг главного процесса
/*
* Fast Fourier Transform.
* Реализация параллельного алгоритма.
* npocs - количество процессов
* myrank - номер текущего процесса
* data - 2-мерный массив (n * 4)
*   n - размер исходных данных (количество коэффициентов многочлена)
*   [0] - действительная часть коэффициентов исходного вектора
*   [1] - мнимая часть коэффициентов исходного вектора
*   [2] - полученная действительная часть после преобразования Фурье
*   [3] - полученная мнимая часть после преобразования Фурье
* type = 1 - прямое преобразование, type = -1 - обратное преобразование
*/
void parallel_fft(int nprocs, int myrank, double(*data)[COLUMNS], int n, int type)
{
	// Главный процесс распространяет всем процессам размер исходных данных.
	MPI_Bcast(&n, 1, MPI_INT, ROOT_RANK, MPI_COMM_WORLD);

	int n_local = n / nprocs; // количество данных на один процесс

	// Создаём собственный буфер с частью исходных данных для каждого процесса.
	double(*subdata)[COLUMNS] = malloc(sizeof(*subdata) * n_local);

	// Главный процесс распределяет данные из исходной таблицы между процессами.
	MPI_Scatter(data, n_local * COLUMNS, MPI_DOUBLE, subdata, n_local * COLUMNS, MPI_DOUBLE, ROOT_RANK, MPI_COMM_WORLD);

	//printf("[%d proc]: Getting unit (%d * %d) with indices (%d..%d).\n",
	//	myrank, n_local, COLUMNS, myrank * n_local, (myrank + 1) * n_local - 1);
	//printf("[%d proc]: My subdata:\n", myrank);
	//for (int i = 0; i < n_local; i++) {
	//	printf("[%d proc]:  [%d]-row: ", myrank, i);
	//	for (int j = 0; j < COLUMNS; j++) {
	//		printf("%9.6lf ", subdata[i][j]);
	//	}
	//	printf("\n");
	//}

	int k, i, j, j_global;
	double factor, angle;
	double complex yk, wnk; // k-й коэффициент, значение k-го корня n-й степени из единицы
	double complex even, odd; // посчитанные чётный и нечётный коэффициенты

	// глобальные суммы действительной и мнимой частей для четных коэффициентов
	double even_real_sum, even_imaginary_sum;
	// глобальные суммы действительной и мнимой частей для нечётных коэффициентов
	double odd_real_sum, odd_imaginary_sum;

	// массивы частичных сумм, в которые главный процесс будет получать локальные суммы от процессов
	double* even_real_sum_parts, * even_imaginary_sum_parts;
	double* odd_real_sum_parts, * odd_imaginary_sum_parts;

	if (myrank == ROOT_RANK) {
		even_real_sum_parts = (double*)malloc(sizeof(double) * nprocs);
		even_imaginary_sum_parts = (double*)malloc(sizeof(double) * nprocs);
		odd_real_sum_parts = (double*)malloc(sizeof(double) * nprocs);
		odd_imaginary_sum_parts = (double*)malloc(sizeof(double) * nprocs);
	}

	// локальные суммы у каждого процесса
	double even_real_local_sum, even_imaginary_local_sum;
	double odd_real_local_sum, odd_imaginary_local_sum;

	for (k = 0; k < n / 2; k++) {
		even_real_sum = even_imaginary_sum = 0.0;
		odd_real_sum = odd_imaginary_sum = 0.0;

		even_real_local_sum = even_imaginary_local_sum = 0.0;
		odd_real_local_sum = odd_imaginary_local_sum = 0.0;

		for (i = 0; i < n_local / 2; i++) { // вычисляем ДПФ для чётных и нечётных коэффициентов исходного вектора
			j = 2 * i;
			factor = type * 2 * PI * k / n;

			j_global = myrank * n_local + j; // с учётом смещения относительно локального индекса

			/* Обработка вектора с чётными коэффициентами */
			yk = subdata[j][0] + subdata[j][1] * I;
			angle = j_global * factor;
			wnk = cos(angle) - sin(angle) * I;
			even = yk * wnk;

			/* Обработка вектора с нечётными коэффициентами */
			yk = subdata[j + 1][0] + subdata[j + 1][1] * I;
			angle = (j_global + 1) * factor;
			wnk = cos(angle) - sin(angle) * I;
			odd = yk * wnk;

			/* Cуммирование посчитанных результатов чётной и нечётной части */
			even_real_local_sum += creal(even);
			even_imaginary_local_sum += cimag(even);

			odd_real_local_sum += creal(odd);
			odd_imaginary_local_sum += cimag(odd);
		}

		// Главный процесс собирает частичные суммы в массив.
		MPI_Gather(&even_real_local_sum, 1, MPI_DOUBLE, even_real_sum_parts, 1, MPI_DOUBLE, ROOT_RANK, MPI_COMM_WORLD);
		MPI_Gather(&even_imaginary_local_sum, 1, MPI_DOUBLE, even_imaginary_sum_parts, 1, MPI_DOUBLE, ROOT_RANK, MPI_COMM_WORLD);
		MPI_Gather(&odd_real_local_sum, 1, MPI_DOUBLE, odd_real_sum_parts, 1, MPI_DOUBLE, ROOT_RANK, MPI_COMM_WORLD);
		MPI_Gather(&odd_imaginary_local_sum, 1, MPI_DOUBLE, odd_imaginary_sum_parts, 1, MPI_DOUBLE, ROOT_RANK, MPI_COMM_WORLD);

		if (myrank == ROOT_RANK) {
			for (i = 0; i < nprocs; i++) { // складываем частичные суммы
				even_real_sum += even_real_sum_parts[i];
				even_imaginary_sum += even_imaginary_sum_parts[i];

				odd_real_sum += odd_real_sum_parts[i];
				odd_imaginary_sum += odd_imaginary_sum_parts[i];
			}

			// Суммирование действительных частей чётных и нечётных коэффициентов.
			data[k][2] = even_real_sum + odd_real_sum; // = yk_real_sum для k = 0..n/2-1
			// Суммирование мнимых частей чётных и нечётных коэффициентов.
			data[k][3] = even_imaginary_sum + odd_imaginary_sum; // = yk_imaginary_sum для k = 0..n/2-1

			// "Преобразование бабочки" для второй половины коэффициентов.
			data[k + n / 2][2] = even_real_sum - odd_real_sum; // = yk_real_sum для k = n/2..n-1
			data[k + n / 2][3] = even_imaginary_sum - odd_imaginary_sum; // = yk_imaginary_sum для k = n/2..n-1

			if (type == INVERSE) {
				data[k][2] /= n;
				data[k][3] /= n;
				data[k + n / 2][2] /= n;
				data[k + n / 2][3] /= n;
			}
		}
	}

	free(subdata);

	if (myrank == ROOT_RANK) {
		free(even_real_sum_parts);
		free(even_imaginary_sum_parts);
		free(odd_real_sum_parts);
		free(odd_imaginary_sum_parts);
	}
}

int fprintLog(char* filename, double(*data)[COLUMNS], double(*inverse_data)[COLUMNS], int n, double time)
{
	FILE* outfile = fopen(filename, "wt");
	if (outfile == NULL) {
		printf("Failed to open \'%s'\n", filename);
		return 0;
	}

	fprintf(outfile, "%-5s |  %19s  %19s  |  %16s  %16s\n",
		"i", "X Real", "X Imaginary", "Error Real", "Error Imaginary");
	for (int i = 0; i < n; i++) {
		fprintf(outfile, "%-5d |  ", i);
		fprintf(outfile, "%19.14lf  %19.14lf  |  ", data[i][2], data[i][3]);
		fprintf(outfile, "%16.13lf  %16.13lf\n", data[i][0] - inverse_data[i][2], data[i][1] - inverse_data[i][3]);
	}

	fprintf(outfile, "\n\nTotal Numbers Processed: %d. Time: %lf sec\n\n", n, time);

	fclose(outfile);

	return 1;
}

int main(int argc, char** argv)
{
	if (argc < 2) {
		printf("Add the input file name as the last argument. ");
		printf("For example: mpirun -n 4 %s data_16_numbers.txt\n", argv[0]);
		return 0;
	}

	char* input_filename = argv[1];

	int argc_new = 1;
	char** argv_new = (char**)malloc(sizeof(char*));
	argv_new[0] = strdup(argv[0]);

	int myrank, nprocs, len;
	char name[MPI_MAX_PROCESSOR_NAME];

	MPI_Init(&argc_new, &argv_new);
	MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
	MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
	MPI_Get_processor_name(name, &len);

	if (myrank == ROOT_RANK) {
		printf("Fast Fourier Transform!\n");
	}

	printf("Hello from host %s[%d] %d of %d\n", name, len, myrank, nprocs);

	int i, j, n;
	// Двумерный массив должен иметь связную область памяти, поэтому механизм
	// двухэтапного выделения памяти сначала под столбцы, а потом под строки, не подойдёт.
	double(*data)[COLUMNS], (*inverse_data)[COLUMNS]; // исходные данные для прямого и обратного преобразования

	if (myrank == ROOT_RANK) {
		printf("[%d proc]: Reading data from file '%s'...\n", myrank, input_filename);

		FILE* fin = fopen(input_filename, "rt");
		if (fin == NULL) {
			printf("[%d proc]: Failed to open \'%s'. Exit.\n", myrank, input_filename);
			MPI_Finalize();
			exit(-1);
		}

		fscanf(fin, "%d", &n);

		data = calloc(n, sizeof(*data));
		for (i = 0; i < n; i++) {
			fscanf(fin, "%lf", &data[i][0]);
		}

		fclose(fin);

		printf("[%d proc]: Data (%d * %d) initialized.\n", myrank, n, COLUMNS);

		//for (i = 0; i < n; i++) {
		//	printf("[%d proc]: read [%d]-row: ", myrank, i);
		//	for (j = 0; j < COLUMNS; j++) {
		//		printf("%9.6lf  ", data[i][j]);
		//	}
		//	printf("\n");
		//}
	}

	double start, end;
	double time_elapsed;
	if (myrank == ROOT_RANK) {
		start = MPI_Wtime();
		printf("\n[%d proc]: Start processing direct transform...\n", myrank);
	}

	// Запускаем параллельный алгоритм для прямого преобразования Фурье.
	parallel_fft(nprocs, myrank, data, n, DIRECT);
	MPI_Barrier(MPI_COMM_WORLD); // ждём, пока все процессы завершат работу

	if (myrank == ROOT_RANK) {
		end = MPI_Wtime();
		time_elapsed = end - start;

		printf("[%d proc]: Direct Fast Fourier Transform processed in %lf seconds!\n", myrank, time_elapsed);

		inverse_data = calloc(n, sizeof(*inverse_data));
		for (i = 0; i < n; i++) {
			inverse_data[i][0] = data[i][2];
			inverse_data[i][1] = data[i][3];
		}

		printf("\n[%d proc]: Start processing inverse transform...\n", myrank);
	}

	// Запускаем параллельный алгоритм для обратного преобразования Фурье.
	parallel_fft(nprocs, myrank, inverse_data, n, INVERSE);
	MPI_Barrier(MPI_COMM_WORLD); // ждём, пока все процессы завершат работу

	if (myrank == ROOT_RANK) {
		printf("[%d proc]: Inverse Fast Fourier Transform completed!\n", myrank);

		char filename[40];
		sprintf(filename, "parallel_fft_%d_numbers_%d_procs.txt", n, nprocs);
		if (fprintLog(filename, data, inverse_data, n, time_elapsed)) {
			printf("\n[%d proc]: The logs of parallel algorithm are written to the file '%s'\n", myrank, filename);
		}

		free(data);
		free(inverse_data);
	}

        for (int i = 0; i < argc_new; i++) {
                free(argv_new[i]);
        }
	free(argv_new);

	MPI_Finalize();
	return 0;
}

