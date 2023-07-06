import numpy as np
from matplotlib import pyplot as plt
from utils import get_job_mochine_num
from pylab import mpl

mpl.rcParams['font.sans-serif'] = ['SimHei']


# 保存输出到log


class data_deal:

    def __init__(self, job_num, machine_num):
        self.job_num = job_num
        self.machine_num = machine_num



    # 将每个工件的每道工序的可选机器和加工时间取出来
    def translate(self, line):
        machine, machine_time, process_select_num, process_index = [], [], [], []
        process_num = line[0]  # 每个工件的工序数
        line = line[1:len(line) + 1]
        index = 0
        # 得到每个工序可选机器数的数字的索引和每道工序可选机器数
        for i in range(process_num):
            sig = line[index]
            process_select_num.append(sig)
            process_index.append(index)
            index = index + 1 + 2 * sig
        # 删除可选工序数的位置
        for j in range(process_num):
            del line[process_index[j] - j]
        # 将机器数和加工时间分别加入两个数组中
        for k in range(0, len(line) - 1, 2):
            machine.append(line[k])
            machine_time.append(line[k + 1])
        return machine, machine_time, process_select_num

    # 获取10个工件中最大工序数
    def width_max(self, parameters):
        width = []
        for i in range(self.job_num):
            mac, mactime, sdx = self.translate(parameters[i])
            sigdx = len(mac)
            width.append(sigdx)
        width = max(width)
        return width

    # 将加工机器和加工时间对应放到两列表中
    def cau(self, parameters):
        width = self.width_max(parameters)
        Cmachine, Cmachinetime = np.zeros((self.job_num, width)), np.zeros((self.job_num, width))
        process_mac_num = []
        for i in range(self.job_num):
            mac, mactime, sdx = self.translate(parameters[i])
            process_mac_num.append(sdx)  # 添加每个工件的每道工序可选机器数
            sig = len(mac)
            Cmachine[i, 0:sig] = mac
            Cmachinetime[i, 0:sig] = mactime
        return Cmachine, Cmachinetime, process_mac_num

    # 得到加工时间，加工机器，工件工序集合，工序数，
    def time_mac_job_pro(self,index):
        parameters,_,_ = get_job_mochine_num.read(index)
        machine, machinetime, process_mac_num = self.cau(parameters)
        jobs, tom = [], []
        for i in range(self.job_num):
            tim = []
            for j in range(1, len(process_mac_num[i]) + 1):
                jobs.append(i)
                tim.append(sum(process_mac_num[i][0:j]))
            tom.append(tim)
        return machine, machinetime, process_mac_num, jobs, tom

    # 种群初始化,得到工件工序对应的机器序列和加工时间
    # def initial_pop(self):
    #
    #     T_machine, T_machinetime, process_mac_num, jobs, tom = self.time_mac_job_pro()
    #     n_machine = np.zeros((1, len(jobs)))
    #     n_machinetime = np.zeros((1, len(jobs)))
    #
    #     # 工序打乱
    #     np.random.shuffle(jobs)
    #     index = [0] * self.job_num
    #     for idx, job in enumerate(jobs):
    #         index_machine = process_mac_num[job][index[job]]  # 得到该工件加工到第几个工序可以使用的机器数
    #         index_tom = tom[job][index[job]]  # 该工件累计工序数
    #         _time = T_machinetime[job][index_tom - index_machine:index_tom]
    #         _machine = T_machine[job][index_tom - index_machine:index_tom]
    #         index[job] += 1
    #         # 随机判断，选择哪道工序的哪台机器
    #         if np.random.rand() > self.pi:  # 选择最小加工时间机器
    #             n_machinetime[0, idx] = min(_time)
    #             index_time = np.argwhere(_time == n_machinetime[0, idx])
    #             n_machine[0, idx] = _machine[index_time[0]]
    #         else:
    #             index_time = np.random.randint(0, len(_time), 1)
    #             n_machine[0, idx] = _machine[index_time[0]]
    #             n_machinetime[0, idx] = _time[index_time[0]]
    #     return jobs, n_machine, n_machinetime

    # 得到最大完工时间
    # def caculate(self, job, machine, machine_time):
    #     jobtime = np.zeros((1, self.job_num))
    #     tmm = np.zeros((1, 6))
    #     tmmw = np.zeros((1, 6))
    #     starttime = 0
    #     list_M, list_S, list_W = [], [], []
    #     for i in range(len(job)):
    #         svg, sig = int(job[i]), int(machine[0, i]) - 1
    #         if (jobtime[0, svg] > 0):
    #             starttime = max(jobtime[0, svg], tmm[0, sig])
    #             tmm[0, sig] = starttime + machine_time[0, i]
    #             jobtime[0, svg] = starttime + machine_time[0, i]
    #         if (jobtime[0, svg] == 0):
    #             starttime = tmm[0, sig]
    #             tmm[0, sig] = starttime + machine_time[0, i]
    #             jobtime[0, svg] = starttime + machine_time[0, i]
    #
    #         tmmw[0, sig] += machine_time[0, i]
    #         list_M.append(machine[0, i])
    #         list_S.append(starttime)
    #         list_W.append(machine_time[0, i])
    #
    #     tmax = np.argmax(tmm[0]) + 1  # 结束最晚机器
    #     C_finish = max(tmm[0])  # 最晚完工时间
    #     return C_finish, list_M, list_S, list_W, tmax
    # 读取算例
    # def read(self):
    #     file_name = get_file.file_name
    #     f = open(file_name)
    #     read = f.readlines()
    #     job_machine_time, count = [], 0
    #     for line in read:
    #         tread = line.strip('\n')
    #         if count > 0:
    #             jmt = []
    #             j = 0
    #             while j < len(tread):
    #
    #                 if tread[j] != ' ' and tread[j] != '\t':
    #                     if j + 1 < len(tread) and tread[j] == ' ' or tread[j] == '\t':
    #                         jmt.append(int(tread[j]))
    #                     if j + 1 < len(tread) and tread[j] != ' ' and tread[j] != '\t':
    #                         num = tread[j] + tread[j + 1]
    #                         jmt.append(int(num))
    #                         j += 1
    #                 j += 1
    #             job_machine_time.append(jmt)
    #         count += 1
    #     return job_machine_time
    # def axis(self):
    #     index = ['M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9', 'M10',
    #              'M11', 'M12', 'M13', 'M14', 'M15', 'M16', 'M17', 'M18', 'M19', 'M20']
    #     scale_ls, index_ls = [], []
    #     for i in range(6):
    #         scale_ls.append(i + 1)
    #         index_ls.append(index[i])
    #
    #     return index_ls, scale_ls  # 返回坐标轴信息，按照工件数返回
    #
    # # 画图
    # def draw(self, job, machine, machine_time):  # 画图
    #     C_finish, list_M, list_S, list_W, tmax = self.caculate(job, machine, machine_time)
    #     figure, ax = plt.subplots()
    #     count = np.zeros((1, self.job_num))
    #     for i in range(len(job)):  # 每一道工序画一个小框
    #         count[0, int(job[i]) - 1] += 1
    #         plt.bar(x=list_S[i], bottom=list_M[i], height=0.5, width=list_W[i], orientation="horizontal", color='white',
    #                 edgecolor='black')
    #         plt.text(list_S[i] + list_W[i] / 32, list_M[i], '%.0f' % (job[i] + 1), color='black', fontsize=10,
    #                  weight='bold')
    #     plt.plot([C_finish, C_finish], [0, tmax], c='black', linestyle='-.', label='完工时间=%.1f' % (C_finish))
    #     font1 = {'weight': 'bold', 'size': 22}
    #     plt.xlabel("加工时间", font1)
    #     plt.title("甘特图", font1)
    #     plt.ylabel("机器", font1)
    #
    #     scale_ls, index_ls = self.axis()
    #     plt.yticks(index_ls, scale_ls)
    #     plt.axis([0, C_finish * 1.1, 0, 6 + 1])
    #     plt.tick_params(labelsize=22)
    #     labels = ax.get_xticklabels()
    #     [label.set_fontname('time new roman') for label in labels]
    #     plt.legend(prop={'family': ['STSong'], 'size': 22})
    #     plt.xlabel("加工时间", font1)
    #     plt.show()
# log = open('log.txt',mode='a',encoding='utf-8')
# # 读取算例数据
# file_name = '../test_data/Brandimarte_Data/Text/Mk01.fjs'
# f = open(file_name)
# read = f.readlines()
# job_machine_time , count = [] ,0
# for line in read:
#     if line=='\n':
#         break
#     tread = line.strip('\n')
#     if(count>0):
#         jmt=[]
#         for j in range(len(tread)):
#             if(tread[j]!=' '):
#                 jmt.append(int(tread[j]))
#         job_machine_time.append(jmt)
#     count+=1
# obj = data_deal(len(job_machine_time),job_machine_time)
# jobs,n_machine,n_machinetime = obj.initial_pop()
# obj.draw(jobs,n_machine,n_machinetime)
# # print(job_machine_time,file=log)
# log.close()
