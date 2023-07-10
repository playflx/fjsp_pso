import random
import numpy as np
from matplotlib import pyplot as plt
from get_data import data_deal

class FJSP:

    def __init__(self,job_num,machine_num,pi,parm_data):
        self.job_num = job_num
        self.machine_num = machine_num
        self.pi = pi
        self.T_machine,self.T_machinetime,self.process_mac_num,self.work,self.tom = parm_data[0],parm_data[1],parm_data[2],parm_data[3],parm_data[4]

    def axis(self):
        index = ['M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9', 'M10',
                 'M11', 'M12', 'M13', 'M14', 'M15', 'M16', 'M17', 'M18', 'M19', 'M20']
        scale_ls, index_ls = [], []
        for i in range(self.machine_num):
            scale_ls.append(i + 1)
            index_ls.append(index[i])

        return index_ls, scale_ls  # 返回坐标轴信息，按照工件数返回

    def creat_jobs(self):
        initial_a = random.sample(range(4*len(self.work)),len(self.work))
        index_jobs = np.array(initial_a).argsort()
        jobs=[]
        for i in range(len(self.work)):
            jobs.append(self.work[index_jobs[i]])
        jobs=np.array(jobs).reshape(1,len(self.work))

        n_machine = np.zeros((1, jobs.shape[1]))
        n_machinetime = np.zeros((1, jobs.shape[1]))
        index = [0] * self.job_num
        machine = [0] * self.machine_num
        for idx, job in enumerate(jobs[0]):
            job=int(job)
            index_machine = self.process_mac_num[job][index[job]]  # 得到该工件加工到第几个工序可以使用的机器数
            index_tom = self.tom[job][index[job]]  # 该工件累计工序数
            high=index_tom
            low=index_tom-index_machine
            _time = self.T_machinetime[job,low:high]
            _machine=self.T_machine[job,low:high]
            index[job] += 1
            if idx < int(jobs.shape[1]*0.8):
                ma,ma_ind,mt=0,0,float('inf')
                _time=_time.tolist()
                _machine=_machine.tolist()
                for ind,mach in enumerate(_machine):
                    if _time[ind] + machine[int(mach)-1] < mt:
                        mt = _time[ind] + machine[int(mach)-1]
                        ma = int(mach)
                        ma_ind = ind
                machine[ma-1]+=_time[ma_ind]
                n_machine[0,idx] = ma
                n_machinetime[0,idx] = _time[ma_ind]
            else:
            # 随机判断，选择哪道工序的哪台机器
            #可以使用迪杰斯特拉算法找到初始最优解------------------------------------------------------------------------------------------------------------
                # if np.random.rand() > self.pi:  # 选择最小加工时间机器
                #     n_machinetime[0, idx] = min(_time)
                #     index_time = np.argwhere(_time == n_machinetime[0, idx])
                #     n_machine[0, idx] = _machine[index_time[0]]
                # else:
                    index_time = np.random.randint(0, len(_time), 1)
                    n_machine[0, idx] = _machine[index_time[0]]
                    n_machinetime[0, idx] = _time[index_time[0]]

        return jobs, n_machine, n_machinetime,initial_a

    # _time = self.T_machinetime[job][(index_tom - index_machine):index_tom]
    # _machine = self.T_machine[job][(index_tom - index_machine):index_tom]
    # 根据新的工序，返回新的加工时间和加工机器集合
    # def to_bact_MT(self,jobs):
    #     n_machine = np.zeros((1, len(jobs)))
    #     n_machinetime = np.zeros((1, len(jobs)))
    #     index = [0] * self.job_num
    #     for idx, job in enumerate(jobs):
    #         index_machine = self.process_mac_num[job][index[job]]  # 得到该工件加工到第几个工序可以使用的机器数
    #         index_tom = self.tom[job][index[job]]  # 该工件累计工序数
    #         _time = self.T_machinetime[job][index_tom - index_machine:index_tom]
    #         _machine = self.T_machine[job][index_tom - index_machine:index_tom]
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
    #     return n_machine,n_machinetime

    def caculate(self,job,machine,machine_time):
        job_time=np.zeros((1,self.job_num))
        tmac_time=np.zeros((1,self.machine_num))
        starttime=0

        list_M,list_S,list_W = [],[],[]
        for i in range(job.shape[1]):
            job_index,mac_index=int(job[0,i]),int(machine[0,i])-1
            if(job_time[0,job_index]>0):
                starttime = max(job_time[0,job_index],tmac_time[0,mac_index])
                tmac_time[0,mac_index]=starttime+machine_time[0,i]
                job_time[0,job_index]=starttime+machine_time[0,i]
            if(job_time[0,job_index]==0):
                starttime=tmac_time[0,mac_index]
                tmac_time[0,mac_index]=starttime+machine_time[0,i]
                job_time[0,job_index]=starttime+machine_time[0,i]
            list_M.append(machine[0,i])
            list_S.append(starttime)
            list_W.append(machine_time[0,i])

        tamx = np.argmax(tmac_time[0])+1  # 结束最晚机器
        C_finish = max(tmac_time[0])  # 最大完工时间
        return C_finish,list_M,list_S,list_W,tamx

    def draw(self,job,machine,machine_time,index): #画图
        C_finish,list_M,list_S,list_W,tmax=self.caculate(job,machine,machine_time)
        figure,ax=plt.subplots()
        count=np.zeros((1,self.job_num))
        for i in range(job.shape[1]): # 每一道工序画一个小框
            count[0,int(job[0,i])] += 1
            plt.bar(x=list_S[i],bottom=list_M[i],height=0.5,width=list_W[i],orientation="horizontal",color='white',edgecolor='black')
            plt.text(list_S[i]+list_W[i]/32,list_M[i],'%.0f' % (job[0,i]+1),color='black',fontsize=10,weight='bold')
        plt.plot([C_finish,C_finish],[0,tmax],c='black',linestyle='-.',label='完工时间=%.1f'%(C_finish))
        font1={'weight':'bold','size':22}
        # plt.xlabel("加工时间",font1)
        plt.title("甘特图MK0{}".format(index),font1)
        plt.ylabel("机器",font1)

        scale_ls,index_ls=self.axis()
        plt.yticks(index_ls,scale_ls)
        plt.axis([0,C_finish*1.1,0,self.machine_num+1])
        plt.tick_params(labelsize=22)
        labels=ax.get_xticklabels()
        [label.set_fontname('FangSong')for label in labels]
        plt.legend(prop={'family':['STSong'],'size':16})
        # plt.xlabel("加工时间",font1)
        plt.savefig('./images/pso'+str(index)+'.png',)
        plt.show()

# file_name = '../test_data/Brandimarte_Data/Text/Mk01.fjs'
# f = open(file_name)
# read = f.readlines()
# job_machine_time , count = [] ,0
# job_num,machine_num = 0,0
# for line in read:
#     if line=='\n':
#         break
#     tread = line.strip('\n')
#     if(count==0):
#         jmt=[]
#         index=0
#         for j in range(len(tread)-1):
#             if(tread[j]==' '):
#                 jmt.append(int(tread[index:j]))
#                 index=j
#         job_num=int(jmt[0])
#         machine_num=int(jmt[1])
#     if(count>0):
#         jmt=[]
#         for j in range(len(tread)):
#             if(tread[j]!=' '):
#                 jmt.append(int(tread[j]))
#         job_machine_time.append(jmt)
#     count+=1
# object_1=data_deal.data_deal(job_num,job_machine_time)
# T_machine,T_machinetime,process_mac_num,jobs,tom = object_1.time_mac_job_pro()
# param=[T_machine,T_machinetime,process_mac_num,jobs,tom]
#
# object_2=FJSP(job_num,machine_num,0.5,param)
# job,machine,machine_time=object_2.creat_jobs()
# object_2.draw(job,machine,machine_time)