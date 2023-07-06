from utils import get_file
def read(index):
    # file_name = get_file.file_name
    file_name = get_file.file_name+str(index)+'.fjs'
    # file_name = './test_data/Brandimarte_Data/Text/Mk01.fjs'
    f = open(file_name)
    read = f.readlines()
    job_machine_time, count = [], 0
    job_num,machine_num = 0,0
    for line in read:
        tread = line.strip('\n')
        if len(tread)==0:break
        if count > 0:
            jmt = []
            j = 0
            while j < len(tread):
                if tread[j] != ' ' and tread[j] != '\t':
                    if j+1 == len(tread):
                        jmt.append(int(tread[j]))
                    if j + 1 < len(tread) and (tread[j+1] == ' ' or tread[j+1] == '\t'):
                        jmt.append(int(tread[j]))
                    if j + 1 < len(tread) and tread[j+1] != ' ' and tread[j+1] != '\t':
                        num = tread[j] + tread[j + 1]
                        jmt.append(int(num))
                        j += 1
                j += 1
            job_machine_time.append(jmt)
        # 文件第一行，取机器数和工件数
        else:
            jm_num = []
            j = 0
            while j < len(tread):
                if len(jm_num) >= 2:
                    break
                if tread[j] != ' ' and tread[j] != '\t':
                    if j + 1 < len(tread) and (tread[j+1] == ' ' or tread[j+1] == '\t') :
                        jm_num.append(int(tread[j]))
                    if j + 1 < len(tread) and tread[j+1] != ' ' and tread[j+1] != '\t':
                        num = tread[j] + tread[j + 1]
                        jm_num.append(int(num))
                        j += 1

                j += 1
            job_num=jm_num[0]
            machine_num=jm_num[1]
        count += 1
    return job_machine_time,job_num,machine_num

# print(a)
# print(b)
# print(c)