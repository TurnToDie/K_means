# K_means++
K means++ algorithm (clustering) \n
Основные действия происходят в функции fit класса K_means.
Вначале за центры кластеров выбираются наиболее удаленные друг от друга точки.
Шаг 1: по матрице расстояний определяются две наиболее удаленных точки и заносятся в массив цетров кластеров. Установить k = 2
Шаг 2: по матрице расстояний определяется наиболее удаленная точка от точек уже занесенных в массив цетров кластеров. 
Увеличить k. Пока k != заданному числу кластеров выполнять Шаг 2
Далее происходит корректировка центров по формулам. 
В итоге выполнения функции fit получим центры кластеров
С помощью функции predict можем определять в какой кластер попадает образ.
До кучи написана визуализация для двумерного случая
