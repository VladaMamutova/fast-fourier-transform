# Fast Fourier Transform

В проекте представлены 2 реализации алгоритма быстрого преобразования Фурье:
- последовательный;
- параллельный с использованием библиотеки MPI.

## Компиляция
```sh
gcc generate_data.c -o generate_data
gcc serial_fast_fourier_transform.c -o serial_fast_fourier_transform -lm
mpicc parallel_fast_fourier_transform.c -o parallel_fast_fourier_transform -lm
```

## Генерация данных
```sh
# генерируем 1024 коэффициента и сохраняем в файл "data_1024_numbers.txt"
./generate_data 1024 data_1024_numbers.txt
```

## Запуск последовательного алгоритма
```sh
# запуск с исходными данными из файла "data_1024_numbers.txt"
./serial_fast_fourier_transform "data_1024_numbers.txt"
```

## Запуск параллельного алгоритма
```sh
# запуск на 4 процесса с исходными данными из файла "data_1024_numbers.txt"
mpirun -np 4 ./parallel_fast_fourier_transform "data_1024_numbers.txt"
```

## Запуск параллельного алгоритма на кластере
```sh
# запуск задачи на 4 процесса с исходными данными из файла "data_1024_numbers.txt"
sbatch job.sh 4 data_1024_numbers.txt
```
