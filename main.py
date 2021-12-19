# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
import sys
from sklearn.model_selection import train_test_split

trainErrAvg = [0]* 8
testErrAvg = [0] * 8






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
    label = []

    for line in file:
        splitedLine = line.split()

        points.append(Point((float)(splitedLine[0]),(float)(splitedLine[1])))
        label.append((int)(splitedLine[2]))
    n = len(label)

    rulesWeight = [0] * (int)((n * n-1))


    return points,label,rulesWeight

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
    # rulesErr = [0] * len(H)


    for i in range(numOfRules):
        ruleSum = claculate_rule_sum(points,label,pointsWeight,H[i])
        # rulesErr[i] = ruleSum
        if ruleSum<min:
            min = ruleSum
            minRuleIndex = i
    return min,minRuleIndex


def update_point_weight(bestRuleWeight,pointsWeight,y_train,rule,x_train):

    for i in range(len(pointsWeight)):
        tempPointLabel=point_label_from_rule(rule,x_train[i])


        pointsWeight[i] = (pointsWeight[i]* np.exp(-(bestRuleWeight*tempPointLabel*y_train[i])))
    pointWeightSum = sum(pointsWeight)
    for i in range(len(pointsWeight)):
        pointsWeight[i] = pointsWeight[i]/pointWeightSum


def calculate_err(points,labels,combinationOfRules,combinationOfRulesWeight):
    totalErr = 0
    for i in range(len(points)):
        sum = 0
        for j in range(len(combinationOfRules)):
            estPointLabel = point_label_from_rule(combinationOfRules[j],points[i])
            sum+=estPointLabel*combinationOfRulesWeight[j]
        if sum<0:
            if labels[i]!=-1:
                totalErr+=1
        else:
            if labels[i]!=1:
                totalErr+=1

    return totalErr/len(points)






def find_error_test_and_train(bestRules,bestRulesWeight,x_train,y_train,x_test,y_test):
    combinationOfRules = []
    combinationOfRulesWeight = []

    for i in range(len(bestRules)):
        combinationOfRules.append(bestRules[i])
        combinationOfRulesWeight.append(bestRulesWeight[i])
        testErr = calculate_err(x_test,y_test,combinationOfRules,combinationOfRulesWeight)
        trainErr = calculate_err(x_train,y_train,combinationOfRules,combinationOfRulesWeight)
        trainErrAvg[i]+=trainErr
        testErrAvg[i]+=testErr


def print_res(iteration):
    for i in range(len(trainErrAvg)):
        print('combine ',(i+1),' rules, train error avg:',trainErrAvg[i]/iteration)
    print('\n\n\n')
    for i in range(len(testErrAvg)):
        print('combine ',(i+1),' rules, test error avg:',testErrAvg[i]/iteration)

    print('\n\n\n\n\n\n*************************************************\n')
    for i in range(len(trainErrAvg)):
        print('combine ',(i+1),' rules, train error avg:',trainErrAvg[i]/iteration)
        print('combine ',(i+1),' rules, test error avg:',testErrAvg[i]/iteration)
        print('\n')

def adaboost(k,path):
    points,label,rulesWeight = init_data(path)
    x_train,x_test,y_train,y_test = train_test_split(points,label ,test_size=0.5)
    pointsWeight = [1/len(x_train)] * len(x_train)
    bestRules = []
    bestRulesWeight = []
    H = []
    H = init_rules(x_train)


    for i in range(k):
        min,minRuleIndex= find_best_rule( x_train,y_train,pointsWeight,H)
        bestRules.append(H[minRuleIndex])
        newWeight = 0.5 * np.log((1.0-min)/min)
        bestRulesWeight.append(newWeight)
        # rulesWeight[minRuleIndex] = newWeight
        update_point_weight(newWeight,pointsWeight,y_train,H[minRuleIndex],x_train)

    find_error_test_and_train(bestRules,bestRulesWeight,x_train,y_train,x_test,y_test)






   # find_error_test_and_train(bestRules,bestRulesWeight,x_train,y_train,x_test,y_test)



def main():
    iteration = 100
    for i in range(iteration):
        adaboost(8, "rectangle.txt")

    print_res(iteration)

    # print(np.log((1.0 - 0.7) / 0.7))








if __name__ == '__main__':
   main()

