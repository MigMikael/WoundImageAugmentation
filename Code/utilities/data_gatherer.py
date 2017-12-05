import os
path = 'C:\\Users\\Mig\\Documents\\Thesis\\data_set'
arr = os.listdir(path)
data = []
for i in arr:
    if i != 'others':
        file_list = os.listdir(path + '\\' + i)
        for f in file_list:
            if f.endswith('.txt'):
                with open(path + '\\' + i + '\\' + f, 'r') as ins:
                    for line in ins:
                        data.append(line)
                #print(i + ' | ' + f)
        #print()
print(len(data))

data_path = 'C:\\Users\\Mig\\WoundImageAugmentation\\Data\\'
count = 1
for line in data:
    if count <= 54:
        with open(data_path + 'test-data.txt', 'a') as outfile:
            outfile.write('|no ' + str(count) + line)
    elif count <= 108:
        with open(data_path + 'evaluate-data.txt', 'a') as outfile:
            outfile.write('|no ' + str(count) + line)
    else:
        with open(data_path + 'train-data.txt', 'a') as outfile:
            outfile.write('|no ' + str(count) + line)
    count += 1
print(count)
