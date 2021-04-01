import numpy as np 
import random
import matplotlib.pyplot as plt

def swaprand(arr):
	arr1 = []
	arr2 = []
	for i in range(8):
		x1 = random.randint(0,99)
		y1 = random.randint(0,149)
		x2 = random.randint(0,99)
		y2 = random.randint(0,149)
		temp = arr[x1,y1]
		arr[x1,y1] = arr[x2,y2]
		arr[x2,y2] = temp

def modifyfirst(arr, i, j):
    first = [(i-1,j-1), (i-1,j), (i-1,j+1), (i,j-1), (i,j+1), (i+1,j-1), (i+1,j), (i+1,j+1)]
    unwanted =[]
    for x in first:
        if x[0]<0 or x[1]<0 or x[0]>99 or x[1]>149:
            unwanted.append(x)
    first = [x for x in first if x not in unwanted]
    for ele in first:
    	if arr[ele[0],ele[1]] ==0:
    		arr[ele[0], ele[1]] = random.choices([1,0],[0.25,0.75])[0]

def modifysecond(arr,i,j):
    second = [(i-2,j-2), (i-2,j-1), (i-2,j), (i-2,j+1), (i-2,j+2), (i-1,j-2), (i-1,j+2), (i,j-2), (i,j+2), (i+1,j-2), (i+1,j+2),(i+2,j-2), (i+2,j-1), (i+2,j), (i+2,j+1), (i+2,j+2)]
    unwanted =[]
    for x in second:
        if x[0]<0 or x[1]<0 or x[0]>99 or x[1]>149:
            unwanted.append(x)
    second = [x for x in second if x not in unwanted]
    for ele in second:
        if arr[ele[0],ele[1]] ==0:
            arr[ele[0], ele[1]] = random.choices([1,0],[0.08,0.92])[0]


arr = np.zeros((100,150), dtype =int)
arr[50, 75] = 1
left = True
posof1s = []
iterations = 0
ones  = [1]
onesadded = [0]
while left:
	left = False
	swaprand(arr)
	for i in range(100):
		for j in range(150):
			if arr[i,j]==1:
				posof1s.append((i,j))
			else :
				left = True
	for pos in posof1s:
		modifyfirst(arr, pos[0], pos[1])
		modifysecond(arr, pos[0], pos[1])
	iterations+=1
	ones.append(len(set(posof1s)))
	onesadded.append(ones[-1]-ones[-2])
	
x = list(range(iterations+1))
plt.subplot(2,1,1)
plt.plot(x,ones)
plt.xlabel("No.of iterations")
plt.ylabel("No. of ones in the matrix")

plt.subplot(2,1,2)
plt.plot(x,onesadded)
plt.xlabel("No. of iterations")
plt.ylabel("No. of ones added")

plt.show()

print("Peak value in plot 2 : %s"%max(onesadded))