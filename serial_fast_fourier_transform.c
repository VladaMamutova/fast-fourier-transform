#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>
#include <time.h>

#define PI 3.141592653589
#define N 655356 // = 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536
#define DIRECT 1 // прямое преобразование
#define INVERSE -1 // обратное преобразование
#define COLUMNS 4 // [0] - действительная часть, [1] - мнимая часть, [2] - действительная часть результата, [3] - мнимая часть результата
#define NUMBER_IS_2_POW_K(x)  (x && !(x & (x - 1)))

/*
* Fast Fourier Transform
* data - 2-мерный массив (n * 4)
* n - размер исходных данных (количество коэффициентов многочлена)
* [0] - действительная часть коэффициентов исходного вектора
* [1] - мнимая часть коэффициентов исходного вектора
* [2] - полученная действительная часть после преобразования Фурье
* [3] - полученная мнимая часть после преобразования Фурье
* type = 1 - прямое преобразование, type = -1 - обратное преобразование
*/
int serial_fft(double** data, int n, int type)
{
    if (n > N || n < 2 || !NUMBER_IS_2_POW_K(n)) {
        printf("'%d' is invalid value for 'n'. It must be a power of two and not exceed %d.\n", n, N);
        return 0;
    }

    if (type != DIRECT && type != INVERSE) {
        printf("'%d' is invalid value for 'type'. It must be 1 for direct transformation or -1 for inverse transformation.\n", type);
        return 0;
    }

    int k, i, j;
    double factor, angle;
    double complex yk, wnk; // k-й коэффициент, значение k-го корня n-й степени из единицы
    double complex even, odd; // посчитанные чётный и нечётный коэффициенты

    // сумма действительной и мнимой частей для четных коэффициентов
    double even_real_sum, even_imaginary_sum;
    // сумма действительной и мнимой частей для нечётных коэффициентов
    double odd_real_sum, odd_imaginary_sum;

    for (k = 0; k < n / 2; k++) {
        even_real_sum = even_imaginary_sum = 0.0;
        odd_real_sum = odd_imaginary_sum = 0.0;

        for (i = 0; i < n / 2; i++) { // вычисляем ДПФ для чётных и нечётных коэффициентов исходного вектора
            j = 2 * i;
            factor = type * 2 * PI * k / n;

            /* Обработка вектора с чётными коэффициентами */
            yk = data[j][0] + data[j][1] * I;
            angle = j * factor;
            wnk = cos(angle) - sin(angle) * I;
            even = yk * wnk;

            /* Обработка вектора с нечётными коэффициентами */
            yk = data[j + 1][0] + data[j + 1][1] * I;
            angle = (j + 1) * factor;
            wnk = cos(angle) - sin(angle) * I;
            odd = yk * wnk;

            // Суммирование посчитанных результатов чётной и нечётной части.
            even_real_sum += creal(even); // действительная часть чётных
            even_imaginary_sum += cimag(even); // мнимая часть чётных

            odd_real_sum += creal(odd); // действительная часть нечётных
            odd_imaginary_sum += cimag(odd); // мнимая часть нечётных
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

    return 1;
}

int fprintLog(char* filename, double** data, double** inverseData, int n, double time)
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
        fprintf(outfile, "%16.13lf  %16.13lf\n", data[i][0] - inverseData[i][2], data[i][1] - inverseData[i][3]);
    }

    fprintf(outfile, "\n\nTotal Numbers Processed: %d. Time: %lf sec\n\n", n, time);

    fclose(outfile);

    return 1;
}

int test_serial_fft(double** data, int n, char* output_filename)
{
    int success;
    clock_t start, end;
    double time_elapsed;
    double** inverse_data; // данные для обратного преобразования Фурье

    printf("Fast Fourier Transform!\n");
    printf("Start processing serial algorithm...\n");

    start = clock();
    success = serial_fft(data, n, DIRECT); // последовательный алгоритм для прямого преобразования Фурье
    if (!success) return 0;

    end = clock();
    time_elapsed = ((double)end - start) / CLOCKS_PER_SEC;
    printf("Direct Fast Fourier Transform completed in %lf sec!\n", time_elapsed);

    inverse_data = (double**)malloc(sizeof(double*) * n);
    for (int i = 0; i < n; i++) {
        inverse_data[i] = (double*)calloc(COLUMNS, sizeof(double));
    }

    for (int i = 0; i < n; i++) {
        inverse_data[i][0] = data[i][2];
        inverse_data[i][1] = data[i][3];
    }

    printf("\nStart processing inverse transform...\n");
    serial_fft(inverse_data, n, INVERSE); // последовательный алгоритм для обратного преобразования Фурье
    printf("Inverse Fast Fourier Transform completed!\n");

    success = fprintLog(output_filename, data, inverse_data, n, time_elapsed);
    if (success) {
        printf("\nThe logs of serial algorithm are written to the file '%s'\n", output_filename);
    }

    for (int i = 0; i < n; i++) {
        free(inverse_data[i]);
    }
    free(inverse_data);

    return 1;
}

int init_data(double*** data, int* n, char* input_filename) {
    FILE* input_file = fopen(input_filename, "rt");
    if (input_file == NULL) {
        printf("Failed to open \'%s'\n", input_filename);
        return 0;
    }

    fscanf(input_file, "%d", n);

    *data = (double**)malloc(sizeof(double*) * *n);
    for (int i = 0; i < *n; i++) {
        (*data)[i] = (double*)calloc(COLUMNS, sizeof(double));
    }

    for (int i = 0; i < *n; i++) {
        fscanf(input_file, "%lf", &(*data)[i][0]);
    }
    fclose(input_file);

    return 1;
}

int main(int argc, char** argv)
{
    if (argc != 2) {
        printf("Usage: %s <filename>\n", argv[0]);
        printf("<filename> - input file name\n");
        exit(1);
    }

    double** data;
    int n;
    char* input_filename = argv[1];

    int success;
    success = init_data(&data, &n, input_filename);
    if (!success) {
        printf("Failed to init data from file \'%s'\n", input_filename);
        exit(1);
    }

    char output_filename[32];
    sprintf(output_filename, "serial_fft_%d_numbers.txt", n);

    test_serial_fft(data, n, output_filename);

    for (int i = 0; i < n; i++) {
        free(data[i]);
    }
    free(data);

    return 0;
}
