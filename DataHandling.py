import csv






def ImportData(file_dir):
    data = []

    with open(file_dir) as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        reader.__next__()
        for row in reader:
            data.append(row)


    return data



def CleanData(data):
    cleaned_data = []

    for d in data:
        # label sex
        if d[4] == 'male':
            sex = 0
        elif d[4] == 'female':
            sex = 1
        else:
            print('error no gender')
            continue

        # label data without age
        if d[5] is '':
        	age = -1
        else:
        	age = float(d[5])

        # label data without fare
        if d[9] is '':
            fare = -1
        else:
            fare = float(d[9])

        #label cabin
        if d[10] != '':
            cabin = 0
        else:
            cabin = 1

        #label embarked
        if d[11] is 'C':
            embarked = 0
        elif d[11] is 'S':            
            embarked = 1
        elif d[11] is 'Q':            
            embarked = 2
        elif d[11] is '':            
            embarked = 3
        else:
            print('error in embarked')
            continue

        c = [int(d[2]), sex , age , int(d[6]) , int(d[7]), fare, cabin, embarked, int(d[1])]

        cleaned_data.append(c)


    return cleaned_data


def CleanDataTest(data):
    cleaned_data = []
    for d in data:
        # label sex
        if d[3] == 'male':
            sex = 0
        elif d[3] == 'female':
            sex = 1
        else:
            print('error no gender')
            continue


        #label embarked
        if d[10] is 'C':
            embarked = 0
        elif d[10] is 'S':            
            embarked = 1
        elif d[10] is 'Q':            
            embarked = 2
        elif d[10] is '':            
            embarked = 3
        else:
            print('error in embarked')
            continue

        # clean data with no age
        if d[4] is '':
            age = -1;           
        else:
            age = float(d[4])
           

        # label data without fare
        if d[8] is '':
            fare = -1
        else:
            fare = float(d[8])

        #label cabin
        if d[9] != '':
            cabin = 0
        else:
            cabin = 1


        c = [int(d[0]),int(d[1]), sex , age , int(d[5]) , int(d[6]), fare, cabin, embarked ]
        	          
        cleaned_data.append(c)


    return cleaned_data


def WriteResults(file_dir,ids,prediction):
    f = open(file_dir,'w', newline='\n')   
    writer = csv.writer(f)
    writer.writerow(['PassengerId','Survived'])
    for i in range(len(ids)):
        writer.writerow([int(ids[i]),int(prediction[i])])
    
    return


def main():
    train_data_dir = 'Data/train.csv'

    train_data = ImportData(train_data_dir)
    train_data = CleanData(train_data)
    #print(train_data[:5])

    test_data_dir = 'Data/test.csv'

    test_data = ImportData(test_data_dir)
    test_data = CleanDataTest(test_data)
    print(test_data[:5])


    return train_data








if __name__ == '__main__':
     main() 
