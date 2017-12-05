data_path = 'C:\\Users\\Mig\\WoundImageAugmentation\\Data\\'

file_name = 'train-data'
file_extension = '.txt'
version = '-3'

line_count = 1
new_data = []
with open(data_path + file_name+file_extension, 'r') as datafile:
    for line in datafile:
        p = line.split('|')
        #print(p[1])     # no 1
        #print(p[2])     # labels
        #print(p[3])     # features
        value = p[2].split(' ')
        center1 = value[73]
        center2 = value[74]
        center3 = value[75]
        l = "|" + p[1] + "|labels " + center1 + " " + center2 + " " + center3 + " |features "

        piece = p[3].split(' ')
        piece.pop(0)
        #print(len(piece))
        for index, item in enumerate(piece):
            string = ''
            if index % 3 == 0 and index <= len(piece) - 2:
                string = piece[index] + " " + piece[index + 1] + " " + piece[index + 2]
                #print(l + str)e
                new_data.append(l + string + '\n')
            #print(l)
            #print(index)
        print('finish line : ' + str(line_count))
        line_count += 1

write_count = 1
for line in new_data:
    if write_count % 1000 == 0:
        print("write line : " + str(write_count))
    with open(data_path + file_name + version + file_extension, 'a') as outfile:
        outfile.write(line)
    write_count += 1

print('finish')
