# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
import sys
class Point:
    def __init__(self,x,y):
        self.x = x
        self.y = y
    def __str__(self):
        repr  = '('+(str)(self.x)+','+(str)(self.y)+')'
        return repr


def init_data(path):
    file = open(path)
    points = []

    H = []
    label = []

    for line in file:
        splitedLine = line.split()

        points.append(Point((float)(splitedLine[0]),(float)(splitedLine[1])))
        label.append((int)(splitedLine[2]))
    n = len(label)
    pointsWeight = [1/n] * n
    rulesWeight = [0] * (int)((n * n-1)/2)
    H = init_rules(points)

    return points,label,pointsWeight,rulesWeight,H

def init_rules(points):
    numOfPoints = len(points)
    H = []
    for i in range(numOfPoints):
        for j in range(i+1,numOfPoints):
            if i==j:
                continue
            H.append([points[i],points[j],1])
            H.append([points[i],points[j],-1])
    return H
def point_label_from_rule(rule,point):
    d = ((point.x-rule[0].x) * (rule[1].y-rule[0].y)) - ((point.y-rule[0].y)*(rule[1].x-rule[0].x))
    if d<0:
        d=-1
    else:
        d=1
    if(rule[2]==1):
        return d
    return (-d)





def claculate_rule_sum(points,label,pointsWeight,rule):
    ruleSum = 0
    numOfPoints = len(points)
    for i in range(numOfPoints):
        currentPointLabel = point_label_from_rule(rule,points[i])
        if not currentPointLabel==label[i]:
            ruleSum+=pointsWeight[i]
    return ruleSum





def find_best_rule(points,label,pointsWeight,H):
    numOfRules = len(H)
    min = sys.maxsize
    minRuleIndex = 0
    rulesErr = [0] * len(H)


    for i in range(numOfRules):
        ruleSum = claculate_rule_sum(points,label,pointsWeight,H[i])
        rulesErr[i] = ruleSum
        if ruleSum<min:
            min = ruleSum
            minRuleIndex = i
    return min,minRuleIndex


def update_point_weight(bestRuleWeight,pointsWeight,labels,rule,points):

    for i in range(len(pointsWeight)):
        tempPointLabel=point_label_from_rule(rule,points[i])
        if not tempPointLabel==labels[i]:
            tempPointLabel = (-labels[i])

        pointsWeight[i] = (pointsWeight[i]* np.exp(-(bestRuleWeight*tempPointLabel*labels[i])))
    pointWeightSum = sum(pointsWeight)
    for i in range(len(pointsWeight)):
        pointsWeight[i] = pointsWeight[i]/pointWeightSum


def adaboost(k,path):
    points,label,pointsWeight,rulesWeight,H = init_data(path)
    print(len(H))
    for i in range(k):
        min,minRuleIndex= find_best_rule(points,label,pointsWeight,H)

        newWeight = 0.5 * np.log((1.0-min)/min)
        rulesWeight[minRuleIndex] = newWeight
        update_point_weight(rulesWeight[minRuleIndex],pointsWeight,label,H[minRuleIndex],points)
    for i in range(len(rulesWeight)):
        if rulesWeight[i]!=0:
            print(rulesWeight[i])



def main():
    adaboost(2,"rectangle.txt")

    # print(np.log((1.0 - 0.7) / 0.7))








if __name__ == '__main__':
   main()

