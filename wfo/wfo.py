import random

import numpy as np

from FJSP.FJSP import FJSP
from get_data.data_deal import data_deal
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']

class wfo():
    def __init__(self,param_fjsp,generation,popsize):
        self.job_num = param_fjsp[0]  # 工件数
        self.machine_num = param_fjsp[1]  # 机器数
        self.pi = param_fjsp[2]  # 选择机器概率
        self.generation = generation  # 迭代次数
        self.popsize = popsize  # 粒子个数


    # 把加工时间编码和加工机器编码转化为对应列表
    def to_MT(self,job,machine,machine_time):
        ma_,maT_,cross_ = [],[],[]
        # 添加工件个数的空列表

        for i in range(self.job_num):
            ma_.append([]),maT_.append([]),cross_.append([])
        for i in range(job.shape[1]):
            sig = int(job[0,i])
            ma_[sig].append(machine[0,i])
            maT_[sig].append(machine_time[0,i])  # 记录每个工件的加工机器和时间
            index = np.random.randint(0,2,1)[0]
            cross_[sig].append(index)  # 随机生成一个0或1的列表，用于后续的机器的均匀交叉
        return ma_,maT_,cross_

    # 列表返回根据新的工序，返回新的加工机器编码和加工时间编码
    def back_MT(self,job,machine,machinetime):
        memory=np.zeros((1,self.job_num),dtype=int)
        m1 ,t1 = np.zeros((1,job.shape[1])),np.zeros((1,job.shape[1]))
        for i in range(job.shape[1]):
            sig=int(job[0,i])
            m1[0,i]=machine[sig][memory[0,sig]]
            t1[0,i]=machinetime[sig][memory[0,sig]]
            memory[0,sig]+=1
        return m1,t1

    # 机器编码和加工时间编码均匀交叉
    def mac_cross(self,Ma_1,Tm_1,Ma_2,Tm_2,cross):
        Mc1,Mc2,Tc1,Tc2 = [],[],[],[]
        for i in range(self.job_num):
            Mc1.append([]),Mc2.append([]),Tc1.append([]),Tc2.append([])
            for j in range(len(cross[i])):
                if(cross[i][j]==0):
                    Mc1[i].append(Ma_1[i][j])
                    Mc2[i].append(Ma_2[i][j])
                    Tc1[i].append(Tm_1[i][j])
                    Tc2[i].append(Tm_2[i][j])
                else:
                    Mc1[i].append(Ma_2[i][j])
                    Mc2[i].append(Ma_1[i][j])
                    Tc1[i].append(Tm_2[i][j])
                    Tc2[i].append(Tm_1[i][j])
        return Mc1,Mc2,Tc1,Tc2

    # 粒子群算法
    def wfo_total(self):
        global obj_fjsp
        obj_datadeal = data_deal(self.job_num,self.machine_num)
        Tmachine, Tmachinetime, process_mac_num, jobs, tom=obj_datadeal.time_mac_job_pro()
        param_data=[Tmachine,Tmachinetime,process_mac_num,jobs,tom]
        obj_fjsp=FJSP(self.job_num,self.machine_num,self.pi,param_data)
        answer,result = [],[]
        job_initial = np.zeros((self.popsize,len(jobs)))
        work_job,work_machine,work_time = np.zeros((self.popsize,len(jobs))),np.zeros((self.popsize,len(jobs))),np.zeros((self.popsize,len(jobs)))
        work_job1, work_machine1, work_time1 = np.zeros((self.popsize, len(jobs))), np.zeros((self.popsize, len(jobs))), np.zeros((self.popsize, len(jobs)))
        for gen in range(self.generation):
            if(gen<1):
                for i in range(self.popsize):
                    job,machine,machine_time,initial_a = obj_fjsp.creat_jobs()
                    C_finish,_,_,_,_=obj_fjsp.caculate(job,machine,machine_time)
                    answer.append(C_finish)
                    work_job[i],work_machine[i],work_time[i] = job[0],machine[0],machine_time[0]
                    job_initial[i]=initial_a
                print('种群初始的最小最大完工时间: %0.f'%(min(answer)))

            index_sort=np.array(answer).argsort()  # 返回完工时间从小到大的位置索引
            work_job1,work_machine1,work_time1=work_job[index_sort],work_machine[index_sort],work_time[index_sort]
            answer1=np.array(answer)[index_sort]
            job_initial1=job_initial[index_sort]
            Alpha=job_initial1[0]  # a狼
            Beta=job_initial1[1]  # b狼
            Delta=job_initial1[2]  # γ狼
            a=2*(1-gen/self.generation)
            #工序编码
            for i in range(3,self.popsize):  # 用最优位置进行工序编码的更新
                job,machine,machine_time=work_job1[i:i+1],work_machine1[i:i+1],work_time1[i:i+1]
                Ma_1,Tm_1,Wcross = self.to_MT(job,machine,machine_time)
                x = job_initial1[i]

                r1 = random.random()  #灰狼算法解的更新
                r2 = random.random()
                A1 = 2 * a * r1 -a
                C1 = 2 * r2
                D_alpha = C1 * Alpha-x
                x1=x-A1 * D_alpha

                r1 = random.random()  # 灰狼算法解的更新
                r2 = random.random()
                A1 = 2 * a * r1 - a
                C1 = 2 * r2
                D_beta = C1 * Beta - x
                x2 = x - A1 * D_beta

                r1 = random.random()  # 灰狼算法解的更新
                r2 = random.random()
                A1 = 2 * a * r1 - a
                C1 = 2 * r2
                D_delta = C1 * Delta - x
                x3 = x - A1 * D_delta

                initial_a = (x1+x2+x3)/3  # 更新公式
                index_work=initial_a.argsort()
                job_new=[]
                for j in range(len(jobs)):
                    job.append(jobs[index_work[j]])
                job=np.array(job).reshape(1,len(jobs))
                machine_new,time_new = self.back_MT(job,Ma_1,Tm_1)
                C_finish,_,_,_,_=obj_fjsp.caculate(job,machine_new,time_new)
                #自我判断这里少了一个判断最大完工时间是否小于已知种群中的某一个完工时间
                work_job[i]=job
                job_initial[i]=initial_a
                work_machine1,work_time1=machine_new[0],time_new[0]
                answer[i]=C_finish


            # 机器编码
            for i in range(0,self.popsize,2):
                job,machine,machine_time=work_job[i:i+1],work_machine[i:i+1],work_time[i:i+1]
                Ma_1,Tm_1,wcross = self.to_MT(job,machine,machine_time)
                job1,machine1,machine_time1=work_job[i+1:i+2],work_machine[i+1:i+2],work_time[i+1:i+2]
                Ma_2,Tm_2,wcross=self.to_MT(job1,machine1,machine_time1)
                Mc1, Mc2, Tc1, Tc2=self.mac_cross(Ma_1,Tm_1,Ma_2,Tm_2,wcross)
                #第一个机器编码
                machine_new,time_new=self.back_MT(job,Mc1,Tc1)
                C_finish,_,_,_,_=obj_fjsp.caculate(job,machine_new,time_new)
                if(C_finish<answer[i]):
                    work_machine[i]=machine_new[0]
                    work_time[i]=time_new[0]
                    answer[i]=C_finish
                # 第二个机器编码
                machine_new1, time_new1 = self.back_MT(job, Mc1, Tc1)
                C_finish, _, _, _, _ = obj_fjsp.caculate(job, machine_new1, time_new1)
                if (C_finish < answer[i]):
                    work_machine[i+1] = machine_new1[0]
                    work_time[i+1] = time_new1[0]
                    answer[i+1] = C_finish
            best_index=answer.index(min(answer))
            gbest=job_initial[best_index]
            result.append(answer[best_index])
        print(answer[best_index],file=log)
        return work_job[best_index],work_machine[best_index],work_time[best_index],result
log = open('log.txt',mode='a',encoding='utf-8')
ho = pso([10,15,0.5],200,100,[1,2,2])
a,b,c,d=ho.pso_total()
job,machine,machine_time = np.array([a]),np.array([b]),np.array([c])
obj_fjsp.draw(job,machine,machine_time)
log.close()