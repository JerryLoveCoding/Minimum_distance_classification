import csv
from sklearn import preprocessing
import numpy as np

filename = 'G:\MLClass\data1.csv'
datas ={}
with open(filename) as f:
    reader = csv.reader(f)#render是节点的data
    i = 1
    for row in reader:
        if i>1:
            datas[str(i-1)] = row
        i = i+1
    print("datas")
    print(datas)#datas1是数据

################################################
def fenlei():
    global datas
    datas1 = {}
    datas0 = {}
    for key in datas:
        row = datas[key]
        if row[6] == "1":
            datas1[key] = row
        elif row[6] == "0":
            datas0[key] = row
    return datas1, datas0

fenlei()
datas1 = {}
datas0 = {}
datastrain1, datastrain0 = fenlei()
print("为1的训练数据样本datastrain1")
print(datastrain1)
print("为0的训练数据样本datastrain0")
print(datastrain0)

#######################行型1数据转化为列型数据#########################
datacol1 = {}#将为1的训练数据样本转化为列型的
dataage = []
datapsa = []
datapv = []
datasuvmax = []
datatrus = []
datadre = []
for key in datastrain1:
    dataage.append(datastrain1[key][0])
    datapsa.append(datastrain1[key][1])
    datapv.append(datastrain1[key][2])
    datasuvmax.append(datastrain1[key][3])
    datatrus.append(datastrain1[key][4])
    datadre.append(datastrain1[key][5])
datacol1["1"] = dataage
datacol1["2"] = datapsa
datacol1["3"] = datapv
datacol1["4"] = datasuvmax
datacol1["5"] = datatrus
datacol1["6"] = datadre
print("列型1数据datacol1")
print(datacol1)
#########################行型0数据转化为列型数据################################
datacol0 = {}#将为0的训练数据样本转化为列型的
dataage = []
datapsa = []
datapv = []
datasuvmax = []
datatrus = []
datadre = []
for key in datastrain0:
    dataage.append(datastrain0[key][0])
    datapsa.append(datastrain0[key][1])
    datapv.append(datastrain0[key][2])
    datasuvmax.append(datastrain0[key][3])
    datatrus.append(datastrain0[key][4])
    datadre.append(datastrain0[key][5])
datacol0["1"] = dataage
datacol0["2"] = datapsa
datacol0["3"] = datapv
datacol0["4"] = datasuvmax
datacol0["5"] = datatrus
datacol0["6"] = datadre
print("列型0数据datacol0")
print(datacol0)
##########对datacol0和datacol1做归一化处理###############
def guiyi(X):
    min_max_scaler = preprocessing.MinMaxScaler()
    X_minMax = min_max_scaler.fit_transform(X)
    return X_minMax
datacol_1 = {}
datacol_0 = {}
for i in range(1, 7):
    #这里用map函数对X执行批处理，即对list中的每个字符串型数字转化为float型
    X = list(map(float, datacol1[str(i)]))
    # 这里将X转化为数组，然后用reshape处理成n行1列的数组，因为min_max_scaler不接受一维数组
    X = np.array(X,dtype='float64').reshape(-1, 1)
    X = guiyi(X)
    X = np.array(X).reshape(-1)
    datacol_1[str(i)] = X.tolist()
for i in range(1, 7):
    Y = list(map(float, datacol0[str(i)]))
    Y = np.array(Y,dtype='float64').reshape(-1, 1)
    Y = guiyi(Y)
    Y = np.array(Y).reshape(-1)
    datacol_0[str(i)] = Y.tolist()
print("处理后的datacol_1")
print(datacol_1)
print("处理后的datacol_0")
print(datacol_0)


###########################求出1和0的训练样本的特征中心########################
#对数据样本求各个特征的平均值
def aver(data):
    sumage = 0
    for i in data["1"]:
        sumage = sumage + float(i)
    sumpsa = 0
    for i in data["2"]:
        sumpsa = sumpsa + float(i)
    sumpv = 0
    for i in data["3"]:
        sumpv = sumpv + float(i)
    sumsuvmax = 0
    for i in data["4"]:
        sumsuvmax = sumsuvmax + float(i)
    sumtrus = 0
    for i in data["5"]:
        sumtrus = sumtrus + float(i)
    sumdre = 0
    for i in data["6"]:
        sumdre = sumdre + float(i)
    averdata = [sumage/len(data["1"]), sumpsa/len(data["2"]), sumpv/len(data["3"]), sumsuvmax/len(data["4"]), sumtrus/len(data["5"]), sumdre/len(data["6"])]
    return averdata

#求biopsy结果为1的样本集和结果为0的样本集各自的特征中心
datacenter1 = aver(datacol_1)
datacenter0 = aver(datacol_0)
print("1的数据样本中心")
print(datacenter1)
print("0的数据样本中心")
print(datacenter0)

########################最小分类算法########################################
#mindistance()的输入是一个list，这个list有6个数据特征，然后计算到datacenter1和datacenter0的距离
#距离谁小就归为那一类的结果
def mindistance(textsample):
    distance1 = (float(textsample[0])-float(datacenter1[0]))**2+(float(textsample[1])-float(datacenter1[1]))**2+(float(textsample[2]) - float(datacenter1[2]))**2 + (float(textsample[3])-float(datacenter1[3]))**2+(float(textsample[4])-float(datacenter1[4]))**2+(float(textsample[5])-float(datacenter1[5]))**2
    distance0 = (float(textsample[0])-float(datacenter0[0]))**2+(float(textsample[1])-float(datacenter0[1]))**2+(float(textsample[2]) - float(datacenter0[2]))**2 + (float(textsample[3])-float(datacenter0[3]))**2+(float(textsample[4])-float(datacenter0[4]))**2+(float(textsample[5])-float(datacenter0[5]))**2
    if distance0>distance1:
        return 0
    elif distance1>distance0:
        return 1

#对datas中的数据进行求解，首先先把其归一化
datascol = {}#将为1的训练数据样本转化为列型的
datasage = []
dataspsa = []
dataspv = []
datassuvmax = []
datastrus = []
datasdre = []
for key in datas:
    datasage.append(datas[key][0])
    dataspsa.append(datas[key][1])
    dataspv.append(datas[key][2])
    datassuvmax.append(datas[key][3])
    datastrus.append(datas[key][4])
    datasdre.append(datas[key][5])
datascol["1"] = datasage
datascol["2"] = dataspsa
datascol["3"] = dataspv
datascol["4"] = datassuvmax
datascol["5"] = datastrus
datascol["6"] = datasdre
print("数据datascol")
print(datascol)
datascol_={}#归一化后的datascol，即全部的datas列型后归一化
for i in range(1, 7):
    #这里用map函数对X执行批处理，即对list中的每个字符串型数字转化为float型
    X = list(map(float, datascol[str(i)]))
    # 这里将X转化为数组，然后用reshape处理成n行1列的数组，因为min_max_scaler不接受一维数组
    X = np.array(X,dtype='float64').reshape(-1, 1)
    X = guiyi(X)
    X = np.array(X).reshape(-1)
    datascol_[str(i)] = X.tolist()
print("归一化的datascol")
print(datascol_)
lenth = len(datascol_["1"])
datas_ = {}#归一化的datas
for i in range(1,lenth):
    datas_[i] = [datascol_["1"][i], datascol_["2"][i],datascol_["3"][i], datascol_["4"][i], datascol_["5"][i], datascol_["6"][i]]
print("对datas归一化的datas_")
print(datas_)

yanzhengdatas = []
for key in datas_:
    result = mindistance(datas_[key])
    yanzhengdatas.append(result)
print("使用全部数据对最小距离分类跑一下的预测结果")
print(yanzhengdatas)
print("结果")
js =0#计数，如果预测结果和真实相等就加1
for i in range(0, len(yanzhengdatas)):
    if int(yanzhengdatas[i]) == int(datas[str(i+1)][6]):
        js += 1
print(float(js/61))