import numpy 
from PIL import Image

def median_filter(data, filter_size):
    temp = []
    indexer = filter_size // 2
    data_final = []
    data_final = numpy.zeros((len(data),len(data[0])))
    for i in range(len(data)):
        for j in range(len(data[0])):
            for z in range(filter_size):
                if i + z - indexer < 0 or i + z - indexer > len(data) - 1:
                    for c in range(filter_size):
                        temp.append(0)
                else:
                    if j + z - indexer < 0 or j + indexer > len(data[0]) - 1:
                        temp.append(0)
                    else:
                        for k in range(filter_size):
                            temp.append(data[i + z - indexer][j + k - indexer])
            temp.sort()
            data_final[i][j] = temp[len(temp) // 2]
            temp = []
    return data_final

#img = Image.open("./img1_3.jpg").convert("L")
#arr = numpy.array(img)
#arr = numpy.zeros((8,8))
#for i in range(8):
#    for j in range(8):
 #       arr[i,j] = abs(i-j)
#print(arr)
#removed_noise = median_filter(arr, 3) 
#print(removed_noise)
#img = Image.fromarray(removed_noise)
#img.show()
