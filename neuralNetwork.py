import numpy as np
import random
import math

class neauralNetwork:
    x = np.array([5.0, 4.0, 3.0])
    w1 = np.zeros((3, 3))
    w2 = np.zeros((2, 3))
    etalon = math.sin(5) + math.sin(4) - math.sin(3)
    n = 0.02 #koef of study
    d = np.array([0.0, 0.0])
    flag = False

    #constructor for neural network
    def __init__(self, weight = None):
        if weight != None:
            k = 0
            self.flag = True
            list_value = weight.split()
            for i in range(3):
                for j in range(3):
                    self.w1[i][j] = float(list_value[k])
                    k+=1
            for i in range(2):
                for j in range(3):
                    self.w2[i][j] = float(list_value[k])
                    k+=1
            self.minim_x = float(list_value[k])
            k+=1
            self.maxim_x = float(list_value[k])
            k+=1
            self.minim_y = float(list_value[k])
            k += 1
            self.maxim_y = float(list_value[k])
            k += 1

    def average(self):
        avg = 0.0
        y = np.arange(0.0, 20.0)
        x = np.zeros((3, 20))
        for i in range(0, 20):
            x[0][i] = a = random.random() * 19
            x[1][i] = b = random.random() * 19
            x[2][i] = c = random.random() * 19
            y[i] = (math.sin(a) + math.sin(b) - math.sin(c))
        self.minim_y = min(y)
        self.maxim_y = max(y)
        self.minim_x = 0.0
        self.maxim_x = 19.0
        for i in range(20):
            for j in range(3):
                x[j][i] = (x[j][i] - self.minim_x) / (self.maxim_x - self.minim_x)
            y[i] = (y[i] - self.minim_y) / (self.maxim_y - self.minim_y)
        return [x, y]

    #calculate sum
    def sum(self, i, number, values=None):
        suma = 0.0
        if number == 1:
            for j in range(3):
                suma += values * self.w1[i][j]
        elif number == 2:
            for j in range(3):
                suma += values[j] * self.w2[i][j]
        elif number == 3:
            for j in range(3):
                suma += values[j] * self.w1[i][j]
        return suma

    #study neural network
    def study(self):
        if self.flag != True:
            return

        x, y = self.average()
        self.x = x
        self.etalon = y

        #delta for hidden nodes
        deltaABC = np.array([0.0, 0.0, 0.0])

        #delta for out nodes
        deltaFinal = np.array([0.0, 0.0])

        #final value
        yFinal = np.array([0.0, 0.0])

        #sum for hidden nodes
        sumABC = np.array([0.0, 0.0, 0.0])

        #values for hidden nodes
        yABC = np.array([0.0, 0.0, 0.0])

        #sum for out nodes
        sumFinal = np.array([0.0, 0.0])

        for _ in range(20):
            k = 10000
            while k != 0:
                for i in range(3):
                    sumABC[i] = self.sum(i, 3, x[:,_])
                    yABC[i] = 1 / (1 + math.exp(-sumABC[i]))
                for i in range(2):
                    sumFinal[i] = self.sum(i, 2, yABC)
                    yFinal[i] = 1 / (1 + math.exp(-sumFinal[i]))

                    #delta
                    deltaFinal[i]  = (yFinal[i] - self.etalon[_]) * yFinal[i] * (1 - yFinal[i])
                    self.d[i] = yFinal[i]

                #work up new koef
                for j in range(3):
                    deltaABC[j] = yABC[j] * (1 - yABC[j]) * (deltaFinal[0] * self.w2[0][j] + deltaFinal[1] * self.w2[1][j])
                for i in range(3):
                    for j in range(3):
                        self.w1[i][j] -= (self.n * deltaABC[i] * yABC[i])
                for i in range(2):
                    for j in range(3):
                        self.w2[i][j] -= (self.n * deltaFinal[i] * self.d[i])
                k -= 1
            print((self.d[1] + self.d[0])/2.0, self.etalon[_], sep="\t")
            print(x[:,_])
            print("\n")

    def check(self, x):
        # sum for hidden nodes
        sumABC = np.array([0.0, 0.0, 0.0])
        yABC = np.array([0.0, 0.0, 0.0])

        # sum for out nodes
        sumFinal = np.array([0.0, 0.0])
        yFinal = np.array([0.0, 0.0])

        y = (math.sin(x[0]) + math.sin(x[1]) - math.sin(x[2]))
        print(y)
        for i in range(3):
            x[i] = (x[i] - self.minim_x) / (self.maxim_x - self.minim_x)
        y = (y - self.minim_y) / (self.maxim_y - self.minim_y)
        for i in range(3):
            sumABC[i] = self.sum(i, 3, x)
            yABC[i] = 1 / (1 + math.exp(-sumABC[i]))
        for i in range(2):
            sumFinal[i] = self.sum(i, 2, yABC)
            yFinal[i] = 1 / (1 + math.exp(-sumFinal[i]))

        res = (yFinal[0] + yFinal[1])/2
        print(res, y, self.minim_y, self.maxim_y)


    #save all result in weight matrix
    def save(self):
        file = open("results", "w")
        for i in range(3):
            for j in range(3):
                if j < 2:
                    file.write(str(float(self.w1[i][j])) + " ")
                else :
                    file.write(str(float(self.w1[i][j])) + "\n")
        for i in range(2):
            for j in range(3):
                if j < 2:
                    file.write(str(float(self.w2[i][j])) + " ")
                else:
                    file.write(str(float(self.w2[i][j])) + "\n")
        file.write(str(self.minim_x) + " " + str(self.maxim_x) + "\n")
        file.write(str(self.minim_y) + " " + str(self.maxim_y))
        file.close()