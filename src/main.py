import numpy as np
import matplotlib.pyplot as plt
import random
import GeneSample #生出数据
import gradientDecent #梯度下降法
import AnalicalSolution #解析解
import CG #共轭梯度法
import costFunc
import Polycompute

#该方法用于client输入字符串选择优化拟合的方法：GD 、CG、解析解
def fit_method(M,X_head,T,str):
    W = np.random.rand(M + 1).reshape(-1,1)  # 随机产生参数向量
    flag = int(input("是否需要在cost fuction中加入正则项：'0' 或 '1'"))
    if str == 'GD':
        alpha = float(input("请输入学习率："))
        numOfItem = int(input("请输入迭代次数"))
        if flag == 0 :
            return gradientDecent.gradient_decent_withoutPun(X_head,T,W,alpha,numOfItem)
        else:
            lambd = float(input('请输入惩罚项系数:'))
            return gradientDecent.gradient_decent_withPun(X_head,T,W,alpha,numOfItem,lambd)
    if str == 'AS':
        if flag == 0:
            return AnalicalSolution.analical_solution_withoutPun(X_head,T)
        else:
            lambd = float(input('请输入惩罚项系数:'))
            return AnalicalSolution.analical_solution_withPun(X_head,T,lambd)
    elif str == 'CG':
        if flag == 0:
            return CG.co_gradient_withoutPun(X_head,T,W)
        else:
            lambd = float(input('请输入惩罚项系数:'))
            return CG.co_gradient_withPun(X_head,T,W,lambd)
    else:
        print("没有该方法")
        return

#请见实验报告：实验结果分析部分——2.解析解（含惩罚项）
def choose_lambda_by_AS():

    number_of_tra = 100 #手动设置样本数量 观察cost随labmda变化的曲线
    X,T = GeneSample.gene_sample(0,1,number_of_tra)
    X_head = np.c_[X ** 0, X] #生成样本对应的矩阵X_head
    for i in range(2, 10):
        X_head = np.c_[X_head, X ** i]
    X_test ,T_test = GeneSample.gene_sample(0,1,100)
    X_h_test = np.c_[X_test ** 0, X_test] #生成测试集对应的矩阵X_h_head
    for i in range(2, 10):
        X_h_test = np.c_[X_h_test, X_test ** i]

    Lambd = [pow(10,-i) for i in range(30,0,-1)]
    cost = [] #对于每一个lambda 存储它在测试集上的损失
    for lambd in Lambd:
        W = AnalicalSolution.analical_solution_withPun(X_head,T,lambd)
        Yt = Polycompute.poly_compute(X_h_test,W)
        cost.append(costFunc.cost_func_withoutPun(Yt,T_test))
    print(cost)
    plt.plot(range(-30,0),cost,linestyle = '',marker = '.')
    plt.title('m = 9,number of training data = '+str(number_of_tra))
    plt.show()
    return

# choose_lambda_by_AS() #



#——————————下面区域用于控制阶数、超参数等——————————————
M =9 #控制阶数
numOfSam = 100 #控制训练样本数

X,T = GeneSample.gene_sample(0,1,numOfSam) #生成训练样本
X_head = np.c_[X**0, X]
for i in range(2, M+1):
    X_head = np.c_[X_head, X ** i]  # 生成样本对应的矩阵 也就是样本x中各点的0到m次方组成的矩阵

#根据调用fit_method方法时最后一个参数的不同选择不同的方法（求解析解、梯度下降法、共轭梯度法）
# W,cost,test = fit_method(M,X_head,T,'GD')
# W = fit_method(M,X_head,T,'AS')
W,k = fit_method(M,X_head,T,'CG')



plt.plot(X,T,linestyle = '',marker = '.',label = 'training set') #在图像中标注训练集点
Xfig = np.linspace(0,1,200,endpoint=True).reshape(-1,1)
X_headfig = np.c_[Xfig**0, Xfig]
for i in range(2, M+1):
    X_headfig = np.c_[X_headfig, Xfig ** i] #生成用于描绘训练出的模型的点集
plt.plot(Xfig,np.sin(2*np.pi*Xfig),label = 'sin(2*pi*X)') #先画出sin（2pix）原曲线
plt.plot(Xfig,np.dot(X_headfig,W),label = 'test func') #再给出拟合的曲线
plt.title('number of samples = '+str(numOfSam))
# plt.plot(cost,color = 'r',label = 'cost of training data') #这两行代码分别用于展示训练过程中
# plt.plot(test,color = 'b',label = 'cost of testing data') #训练集和测试集的cost
plt.legend()
plt.show()













