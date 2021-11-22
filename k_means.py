import numpy as np
import os
import copy

from numpy.core.fromnumeric import argmin, argmax
import matplotlib.pyplot as  plt
import itertools

#заполнение матрицы начальных данных
def matrix_completion(V,filename):
    with open(os.path.join(os.getcwd(),filename),'r',encoding='utf-8') as file:
        lines = file.readlines()
        for line in lines:
            line = line.split(',')
            time = []
            for item in line:
                if '(' in item:
                    time += [float(item[1:])]
                if ')' in item:
                    time += [float(item[:item.index(')')])]
                elif '(' not in item and ')' not in item:
                    time += [float(item)]
            V.append(time)
    return V

def saveModel(centers_clasters):
    with open('claster_centers.txt','w',encoding='utf-8') as file:
        for i in centers_clasters:
            s = '('
            for j in i:
                s = s + str(j) + ','
            s = s[:-1] + ')\n'
            file.write(s)

#модель
class K_means:
    def __init__(self,clasters_count=2):
        self.clasters_count = clasters_count
        self.weights = np.ones(self.clasters_count)

    def fit(self,X):
        #процедура заполнения первоначальных центров
        #-------------------------------------------------------------------------------------------------------------------------------------------------------
        #заполняем матрицу расстояний
        n = len(X)
        data = np.ones((n,n))
        i = 1
        max_ = 0
        #после этого for(-a) знаем матрицу расстояний, максимальное расстояние и data[max_ind//n][max_ind%n] - макс норма
        for el in itertools.combinations(np.array(X),2):
            if i % n == 0:
                i += (i // n + 1)
            buff = np.linalg.norm(el[0]-el[1])
            if buff > max_:
                max_ = buff
                max_ind = i
            data[i//n][i%n] = data[i%n][i//n] = buff
            i += 1
        #заносим два наиболее удаленных друг от друга вектора
        self.clasters_centers = [X[max_ind//n],X[max_ind%n]]
        clasters_indexes = [max_ind//n,max_ind%n]
        buff = copy.deepcopy(data[max_ind//n])        
        for i in range(2,self.clasters_count): #это не тяжелый for
            #идея - сумма двух строк матрицы и нахождение макс элемента в ней найдет элемент, наиболее удаленный от двух центров одновременно
            #добавляем только один, остальные уже хранятся в буффере
            buff += data[max_ind%n]
            max_ind = argmax(buff)
            while np.all(np.in1d(X[max_ind],self.clasters_centers)): #если этот элемент уже и так взят центральным. Этот цикл тоже не является тяжелым, все ок
                buff[max_ind] = float('-inf') #делаем его недоступным для добавления (он не добавится, т.к. он становится минимальным, а не максимальным)
                max_ind = argmax(buff)
            self.clasters_centers.append(X[max_ind])
            clasters_indexes.append(max_ind)
        #-------------------------------------------------------------------------------------------------------------------------------------------------------
        #по оставшимся точкам
        #идея - берем точку, смотрим к какому центру она ближе и тот центр двигаем
        self.clasters_centers = np.array(self.clasters_centers)
        for i in range(len(X)):
            if X[i] not in self.clasters_centers:
                #смотрим мин расстояние между след элементом и центрами кластеров
                ind = argmin([data[i][j] for j in clasters_indexes]) #массив в скобках имеет кол-во элементов len(claster_indexes) и argmin укажет на тот центр кластера, который нужно сдвинуть
                self.clasters_centers[ind] = np.divide(self.weights[ind]*self.clasters_centers[ind] + X[i],self.weights[ind] + 1)
                #пересчитываем веса
                self.weights[ind] += 1
        return self.clasters_centers

    def predict(self,X):
        #заполняем словарь кластеров
        claster_dict = dict()
        for i in range(self.clasters_count):
            claster_dict[i] = []
        #относим элементы к кластерам
        for el in X:
            #также высчитываем минимальное расстояние, берем его индекс - это и будет индекс кластера, забиваем словарь
            claster_dict[argmin(np.array([np.linalg.norm(el-self.clasters_centers[i]) for i in range(len(self.clasters_centers))]))].append(el)
        return claster_dict

def visualize(X,clasters_centers):
    plt.title('Представление точек')
    x_1 = []
    y_1 = []
    for el in X:
        x_1.append(el[0])
        y_1.append(el[1])
    plt.scatter(x_1,y_1,marker='o',c='b',edgecolor='b')
    x_1 = []
    y_1 = []
    for el in clasters_centers:
        x_1.append(el[0])
        y_1.append(el[1])
    plt.scatter(x_1,y_1,marker='o',c='r',edgecolor='b')
    plt.grid()
    plt.show()

if __name__ == '__main__':
    num_clasters = 3
    k_means = K_means(num_clasters)
    # X = matrix_completion([],'real_data.txt')
    X = np.loadtxt('data_clustering.txt', delimiter=',')
    # #центрируем выборку
    # X -= np.mean(X,axis=0)
    # #нормируем выборку
    # X = X/np.sum(X,axis=1)[:,None]
    clasters_centers = k_means.fit(X)
    if len(clasters_centers[0]) == 2:
        visualize(X,clasters_centers)
    # если нужно сохранить модель (обученную)
    # saveModel(clasters_centers)
    dict_ = k_means.predict(X)
    # средний диаметр кластера
    avg_diam = np.average([max([np.linalg.norm(el[0]-el[1]) for el in itertools.combinations(dict_[k],2)]) for k in range(num_clasters)])
    
