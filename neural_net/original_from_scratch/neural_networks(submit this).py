# neural network coursework

from math import e, tanh, sqrt
import pytorch
import random
import xlrd
import matplotlib.pyplot as plt
import numpy as np


x = 0

path = 'CWData.xlsx'

data = []

training = []  # 60% of data
validation = []  # 20% of data
testing = []  # 20 % of data

workbook = xlrd.open_workbook(path)
worksheet = workbook.sheet_by_index(1)

#back_prop(epochs, weight_set, extra_weights, bias_set, lp, momentum, test_set, activation, False)

# retrieves data from the worksheet and stores each row as an array in a 2d array
for row in range(1, worksheet.nrows):
    items = worksheet.row_slice(row)
    area = items[0].value
    bfihost = items[1].value
    farl = items[2].value
    fpext = items[3].value
    ldp = items[4].value
    propwet = items[5].value
    rmed = items[6].value
    saar = items[7].value
    flood = items[8].value

    data.append([area, bfihost, farl, fpext, ldp, propwet, rmed, saar, flood])
#print(data)

# splitting the data into training, validation and training sets
length = len(data)

train_val = round(0.6 * length)

for i in range(0, train_val):
    training.append(data[i])

val_test = round(0.8 * length)

for i in range(train_val, val_test):
    validation.append(data[i])

for i in range(val_test, length):
    testing.append(data[i])


# weights from inputs to hidden layer
w0 = .1
w1 = .2
w2 = .3
w3 = .1
w4 = .2
w5 = .3
w6 = .1
w7 = .2

# weights from hidden layer to output
w0o = .1
w1o = .2
w2o = .3
w3o = .1
w4o = .2
w5o = .3
w6o = .1
w7o = .2


#learning parameter
lp = 0.1

bias_1 = 1
bias_2 = 1
bias_3 = 1
bias_4 = 1
bias_5 = 1
bias_6 = 1
bias_7 = 1
bias_8 = 1
bias_9 = 1

activation="tan"

epochs=2000

w0to1= 1
w1to2 = 2
w2to3 = 3
w3to4 = 1
w4to5 = 2
w5to6  = 3
w6to7 = 1

w1to0 = 1
w2to1 = 2
w3to2 = 3
w4to3 = 1
w5to4 = 2
w6to5 = 3
w7to6 = 1

test_set=training

momentum= 0

actual_arr=[]
pred_arr=[]
error_arr = []
epoch_arr = []
actual=[]
pred=[]

counter=1000


# backpropagation algorithm
weight_set = [w0,w0o,w1,w1o,w2,w2o,w3,w3o,w4,w4o,w5,w5o,w6,w6o,w7,w7o]
bias_set = [bias_1, bias_2, bias_3, bias_4, bias_5, bias_6, bias_7, bias_8, bias_9]

extra_weights = [w1to0,w2to1,w3to2,w4to3 ,w5to4,w6to5,w7to6, w0to1, w1to2, w2to3, w3to4, w4to5, w5to6, w6to7]


#msre function
def msre (pred_arr, actual_arr):
    value=0
    top=0
    for index in range(len(pred_arr)):
        top= (pred_arr[index] - actual_arr[index])
        bottom= actual_arr[index]
        value+=(top/bottom)**2

    value*=(1/len(pred_arr))



    return value

#coefficient error
def ce (pred,actual, obs_mean):
    top=0
    for i in range(len(pred)):
        top+=(pred[i] - actual[i])**2
    bottom=0
    for real in actual:
        bottom+= (real -obs_mean)**2
    value =  1 -(top/bottom)

    return  value
# rsqr
def rsqr (actual_values, mean_obsv, pred_values, mean_mod):
    top =0
    for i in range(len(actual_values)):
        top+=(actual_values[i]-mean_obsv) * (pred_values[i]-mean_mod)
    bottom =0
    for i in range(len(actual_values)):
        bottom+=((actual_values[i] - mean_obsv)**2) * ((pred_values[i] - mean_mod)**2)
    value = (top/sqrt(bottom))**2

    return value

# annealing
# p = end parameter
# q = starting parameter
# r = max epochs
# x = epochs so far
def anneal(p,q,r,x):
    if x < r:
        value= p + (q-p)
        value*=(1- ((1)/(1+e**(10-(20*x)/(r)))))
        return  value


#weight decay
def decay(error,lp,epoch, weights, extra_weights):

    weight= weights + extra_weights

    omega = 0.5* sum(np.square(weight))

    upsilon= 1 / (lp*epoch)

    upsilon = 0.1

    decay= error + upsilon * omega

    return decay
# integrates bold driver with the paremeters:
# lp = old learning rate
# old = previous error
# new = current error
# lim = percentage difference between the old and new
# interval = interval of epochs that will change the lp
# epoch = current number of epocs
def bold_driver(lp,old,new,lim,interval,epoch):
    if epoch % interval==0:
        change=(old-new)

        if abs(change)>lim:
            if change > 0:
                return lp*1.05
            elif change< 0:
                return lp*0.7
            elif change==0:
                return lp
        else:
            return lp
    else:
        return lp
def back_prop(epoch, weights, extra_weight, bias, lp, momentum, set, acc, values):
    cofe=-10
    #error=21
    rmse = 7000
    #while (rmse>800):
    rmse = 0
    print("lp: ",lp)
    print("momentum: ", momentum)

    actual_arr = []
    pred_arr = []
    error_arr = []
    epoch_arr = []

    if acc=="tan":
        def activation(tan):
            return tanh(tan)

        def diff(node):
            value = 1 - node**2
            return value


    elif acc=="sigmoid":

        def activation(sig):
            #print(sig)
            value = (1) / (1 + (e ** (sig*-1)))
            #print(value)
            return value

        def diff(node):
            value = (node) * (1 - (node))
            return value

    elif acc=="linear":

        def activation(lin):
            return lin

        def diff(node):
            return 1
    if values:
        for x in range(len(weights)):
            weights[x]= float(input("enter initial weights"))
            print("weight: ",weights[x])

        for y in range(len(extra_weight)):
            extra_weight[y] = float(input("enter extra weights"))
            print("weight: ",extra_weight[y])

        for z in range(len(bias)):
            bias[z] = float(input("enter biases"))
            print("bias: ",bias[z])
    # weights from inputs to hidden layer
    else:
        for x in range(len(weights)):
            weights[x] = round(random.uniform(-2 / len(set), 2 / len(set)), 4)
            print("weight: ", weights[x])

        for y in range(len(extra_weight)):
            extra_weight[y] = round(random.uniform(-2 / len(set), 2 / len(set)), 4)
            print("weight: ", extra_weight[y])

        for z in range(len(bias)):
            bias[z] = round(random.uniform(-2 / len(set), 2 / len(set)), 4)
            print("bias: ", bias[z])
    # weights from inputs to hidden layer
    w0 = weights[0]
    w1 = weights[2]
    w2 = weights[4]
    w3 = weights[6]
    w4 = weights[8]
    w5 = weights[10]
    w6 = weights[12]
    w7 = weights[14]


    # weights from hidden layer to output
    w0o = weights[1]
    w1o = weights[3]
    w2o = weights[5]
    w3o = weights[7]
    w4o = weights[9]
    w5o = weights[11]
    w6o = weights[13]
    w7o = weights[15]


    # extra weights
    w1to0 = extra_weight[0]
    w2to1 = extra_weight[1]
    w3to2 = extra_weight[2]
    w4to3 = extra_weight[3]
    w5to4 = extra_weight[4]
    w6to5 = extra_weight[5]
    w7to6 = extra_weight[6]

    w0to1 = extra_weight[7]
    w1to2 = extra_weight[8]
    w2to3 = extra_weight[9]
    w3to4 = extra_weight[10]
    w4to5 = extra_weight[11]
    w5to6 = extra_weight[12]
    w6to7 = extra_weight[13]


    bias_1 = bias[0]
    bias_2 = bias[1]
    bias_3 = bias[2]
    bias_4 = bias[3]
    bias_5 = bias[4]
    bias_6 = bias[5]
    bias_7 = bias[6]
    bias_8 = bias[7]

    bias_9 = bias[8]


    for x in range(epoch):
        error_sum=0

        for i in range(len(set)):

            weight_sum=0

            val_1= set[i][0]
            val_2 = set[i][1]
            val_3 = set[i][2]
            val_4 = set[i][3]
            val_5 = set[i][4]
            val_6 = set[i][5]
            val_7 = set[i][6]
            val_8 = set[i][7]
            val_9 = set[i][8]

            # forward pass
            sum_val_1= (w0 * val_1) + bias_1 + (w1to0 * val_2)
            u_1 = activation(sum_val_1)
            weight_sum+= (u_1 * w0o)

            sum_val_2 = (w1 * val_2) + bias_2 + (w2to1 * val_3) + (w0to1* val_1)
            u_2 = activation(sum_val_2)
            weight_sum += (u_2 * w1o)

            sum_val_3 = (w2 * val_3) + bias_3 + (w3to2 * val_4) + (w1to2 * val_2)
            u_3 = activation(sum_val_3)
            weight_sum+=(u_3 * w2o)

            sum_val_4 = (w3 * val_4) + bias_4 + (w4to3 * val_5) + (w2to3 * val_3)
            u_4 = activation(sum_val_4)
            weight_sum += (u_4 * w3o)

            sum_val_5 = (w4 * val_5) + bias_5 + (w3to4 * val_6) + (w5to4 * val_4)
            u_5 = activation(sum_val_5)
            weight_sum+= (u_5 * w4o)

            sum_val_6 = (w5 * val_6) + bias_6 + (w6to5 * val_7) + (w4to5 * val_5)
            u_6 = activation(sum_val_6)
            weight_sum += (u_6 * w5o)

            sum_val_7 = (w6 * val_7) + bias_7 + (w7to6 * val_8) + (w5to6 * val_6)
            u_7 = activation(sum_val_7)
            weight_sum+= (u_7 * w6o)

            sum_val_8 = (w7 * val_8) + bias_8 + (w6to7 * val_7)
            u_8 = activation(sum_val_8)
            weight_sum+= (u_8 * w7o)

            weight_sum+=bias_9

            u_output = activation(weight_sum)

            actual_val_9 = val_9 * (469.699 - 0.406) + 0.406
            pred_val_9= u_output * (469.699 - 0.406) + 0.406



            error = (val_9-u_output)

            acc_error=error * ((469.699 - 0.406) + 0.406)

            #error = decay(error,lp,x+1,weights, extra_weights)
            #print(error)

            # backward pass
            u_output_diff = diff(u_output)

            delta_output = (error) * u_output_diff



            u_1_diff = diff(u_1)
            delta_u1 = w0o * u_1_diff * delta_output

            u_2_diff = diff(u_2)
            delta_u2 = w1o * u_2_diff * delta_output

            u_3_diff = diff(u_3)
            delta_u3 = w2o * u_3_diff * delta_output

            u_4_diff = diff(u_4)
            delta_u4 = w3o * u_4_diff * delta_output

            u_5_diff = diff(u_5)
            delta_u5 = w4o * u_5_diff * delta_output

            u_6_diff = diff(u_6)
            delta_u6 = w5o * u_6_diff * delta_output

            u_7_diff = diff(u_7)
            delta_u7 = w6o * u_7_diff * delta_output

            u_8_diff = diff(u_8)
            delta_u8 = w7o * u_8_diff * delta_output

            # update biases and weights

            bias_1 += lp * delta_u1 * 1
            bias_2 += lp * delta_u2 * 1
            bias_3 += lp * delta_u3 * 1
            bias_4 += lp * delta_u4 * 1
            bias_5 += lp * delta_u5 * 1
            bias_6 += lp * delta_u6 * 1
            bias_7 += lp * delta_u7 * 1
            bias_8 += lp * delta_u8 * 1

            bias_9 += lp * delta_output * 1



            w0to1 += lp * delta_u2 * u_2 + (momentum * (w0to1 - (w0to1 + lp * delta_u2 * u_2)))

            w1to2 += lp * delta_u3 * u_3 + (momentum * (w1to2 - (w1to2 + lp * delta_u3 * u_3)))

            w2to3 += lp * delta_u4 * u_4 + (momentum * (w2to3 - (w2to3 + lp * delta_u4 * u_4)))

            w3to4 += lp * delta_u5 * u_5 + (momentum * (w3to4 - (w3to4 + lp * delta_u5 * u_5)))

            w4to5 += lp * delta_u6 * u_6 + (momentum * (w4to5 - (w4to5 + lp * delta_u6 * u_6)))

            w5to6 += lp * delta_u7 * u_7 + (momentum * (w5to6 - (w5to6 + lp * delta_u7 * u_7)))

            w6to7 += lp * delta_u8 * u_8 + (momentum * (w6to7 - (w6to7 + lp * delta_u8 * u_8)))

            w1to0 += lp * delta_u1 * u_1 + (momentum * (w1to0 - (w1to0 + lp * delta_u1 * u_1)))

            w2to1 += lp * delta_u2 * u_2 + (momentum * (w2to1 - (w2to1 + lp * delta_u2 * u_2)))

            w3to2 += lp * delta_u3 * u_3 + (momentum * (w3to2 - (w3to2 + lp * delta_u3 * u_3)))

            w4to3 += lp * delta_u4 * u_4 + (momentum * (w4to3 - (w4to3 + lp * delta_u4 * u_4)))

            w5to4 += lp * delta_u5 * u_5 + (momentum * (w5to4 - (w5to4 + lp * delta_u5 * u_5)))

            w6to5 += lp * delta_u6 * u_6 + (momentum * (w6to5 - (w6to5 + lp * delta_u6 * u_6)))

            w7to6 += lp * delta_u7 * u_7 + (momentum * (w7to6 - (w7to6 + lp * delta_u7 * u_7)))


            w0 += lp * delta_u1 * u_1 + (momentum * (w0 - (w0 + lp * delta_u1 * u_1)))

            w1 += lp * delta_u2 * u_2 + (momentum * (w1 - (w1 + lp * delta_u2 * u_2)))

            w2 += lp * delta_u3 * u_3 + (momentum * (w2 - (w2 + lp * delta_u3 * u_3)))

            w3 += lp * delta_u4 * u_4 + (momentum * (w3 - (w3 + lp * delta_u4 * u_4)))

            w4 += lp * delta_u5 * u_5 + (momentum * (w4 - (w4 + lp * delta_u5 * u_5)))

            w5 += lp * delta_u6 * u_6 + (momentum * (w5 - (w5 + lp * delta_u6 * u_6)))

            w6 += lp * delta_u7 * u_7 + (momentum * (w6 - (w6 + lp * delta_u7 * u_7)))

            w7 += lp * delta_u8 * u_8 + (momentum * (w7 - (w7 + lp * delta_u8 * u_8)))

            w0o += lp * delta_u1 * u_1 + (momentum * (w0o - (w0o + lp * delta_u1 * u_1)))

            w1o += lp * delta_u2 * u_2 + (momentum * (w1o - (w1o + lp * delta_u2 * u_2)))

            w2o += lp * delta_u3 * u_3 + (momentum * (w2o - (w2o + lp * delta_u3 * u_3)))

            w3o += lp * delta_u4 * u_4 + (momentum * (w3o - (w3o + lp * delta_u4 * u_4)))

            w4o += lp * delta_u5 * u_5 + (momentum * (w4o - (w4o + lp * delta_u5 * u_5)))

            w5o += lp * delta_u6 * u_6 + (momentum * (w5o - (w5o + lp * delta_u6 * u_6)))

            w6o += lp * delta_u7 * u_7 + (momentum * (w6o - (w6o + lp * delta_u7 * u_7)))

            w7o += lp * delta_u8 * u_8 + (momentum * (w7o - (w7o + lp * delta_u8 * u_8)))

            rmse+=(acc_error)**2
            


        actual_arr.append(actual_val_9)
        pred_arr.append(pred_val_9)
        error_arr.append(acc_error)
        epoch_arr.append(x)
        #lp = bold_driver(lp, error_arr[x - 1], error_arr[x], 1, 1000, x)


    avg_pred = sum(pred_arr) / len(pred_arr)
    avg_error = sum(error_arr) / len(error_arr)
    avg_actual = sum(actual_arr) / len(actual_arr)
    rootsquare = rsqr(actual_arr, avg_actual, pred_arr, avg_pred)
    meansre= msre(pred_arr, actual_arr)
    print("acc: ", actual_val_9)
    print("pred: ", pred_val_9)
    print("rsqr: ", rootsquare)
    print("msre: ", meansre)

    cofe=ce(pred_arr, actual_arr, avg_actual)
    rmse = sqrt(rmse / len(epoch_arr))
    print("rmse: ",rmse)
    print("ce: ", cofe)

    plt.plot(epoch_arr, error_arr, 'r+')
    plt.ylabel('errors')
    plt.xlabel('epochs')
    plt.show()

back_prop(epochs,weight_set, extra_weights, bias_set,lp,momentum,test_set, activation, False)


'''
#LMS Algorithm done for comparison

w8=1
for i in range(len(testing)):
    for j in range(8):

        bias = 1
        area = training[j][0]
        bfihost= training[j][1]
        farl=training[j][2]
        fpext= training[j][3]
        ldp= training[j][4]
        propwet= training[j][5]
        rmed=training[j][6]
        saar=training[j][7]
        flood=training[j][8]

        error=flood-(w0+(w1*area)+(w2*bfihost) + (w3*farl)+(w4*fpext)
                              +(w5*ldp)+(w6*propwet) + (w7*rmed)+(w8*saar))
        w0=w0+lp*error*bias
        w1=w1+lp*error*area
        w2=w2+lp*error*bfihost
        w3 = w3 + lp * error * farl
        w4 = w4 + lp * error * fpext

        w5 = w5 + lp * error * ldp
        w6 = w6 + lp * error * propwet
        w7 = w7 + lp * error * rmed
        w8 = w8 + lp * error * saar

    #print("Loop: " +str((i+1)))
    error_arr.append(error*-1)
    weight_set.append([w0, w1, w2, w3, w4, w5, w6, w7, w8])
    epoch_arr.append(i)

print(weight_set)

plt.plot(epoch_arr, error_arr, 'r+')
plt.ylabel('errors')
plt.xlabel('epochs')
plt.show()
'''

