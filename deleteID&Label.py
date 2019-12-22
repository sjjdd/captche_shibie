file1 = open('./submission.csv', 'r').readlines()
fileout = open('./.csv', 'w')
fileout.write('ID,label\n')
for line in file1:
    # print('line: ',line)
    # print('line.type: ',type(line))
    # print('len(len):',len(line))
    # print('line[8]:',line[8])
    # print(line[8] is '\n')
    # print('line: ',line.strip())
    # a='ID,label'
    # print(line.strip()  == a)
    if line.strip() != 'ID,label':
        fileout.write(line)
fileout.close()
