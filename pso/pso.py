import random
from utils import get_job_mochine_num
import numpy as np
import config
from FJSP.FJSP import FJSP
from get_data.data_deal import data_deal
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']

class pso():
    def __init__(self,param_fjsp,generation,popsize,param_pso):
        self.job_num = param_fjsp[0]  # 工件数
        self.machine_num = param_fjsp[1]  # 机器数
        self.pi = param_fjsp[2]  # 选择机器概率
        self.generation = generation  # 迭代次数
        self.popsize = popsize  # 粒子个数
        self.W = param_pso[0]
        self.C1 = param_pso[1]
        self.C2 = param_pso[2]

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
    def pso_total(self,index):
        global obj_fjsp
        obj_datadeal = data_deal(self.job_num,self.machine_num)
        Tmachine, Tmachinetime, process_mac_num, jobs, tom=obj_datadeal.time_mac_job_pro(index)
        param_data=[Tmachine,Tmachinetime,process_mac_num,jobs,tom]
        obj_fjsp=FJSP(self.job_num,self.machine_num,self.pi,param_data)
        answer,result = [],[]
        job_initial,pbest = np.zeros((self.popsize,len(jobs))),np.zeros((self.popsize,len(jobs)))
        work_job,work_machine,work_time = np.zeros((self.popsize,len(jobs))),np.zeros((self.popsize,len(jobs))),np.zeros((self.popsize,len(jobs)))
        v = np.zeros((self.popsize,len(jobs)))
        W = self.W
        for gen in range(self.generation):
            if(gen<1):
                for i in range(self.popsize):
                    job,machine,machine_time,initial_a = obj_fjsp.creat_jobs()
                    C_finish,_,_,_,_=obj_fjsp.caculate(job,machine,machine_time)
                    answer.append(C_finish)
                    work_job[i],work_machine[i],work_time[i] = job[0],machine[0],machine_time[0]
                    job_initial[i]=initial_a
                    pbest[i]=initial_a
                best_index=answer.index(min(answer))
                gbest=pbest[best_index]
                print('种群初始的最小最大完工时间: %0.f'%(min(answer)))
            #工序编码
            res = 0
            for i in range(self.popsize):
                job,machine,machine_time=work_job[i:i+1],work_machine[i:i+1],work_time[i:i+1]
                Ma_1,Tm_1,Wcross = self.to_MT(job,machine,machine_time)
                x = job_initial[i]
                v[i] = W * v[i] + self.C1 * random.random() * (pbest[i] - x) \
                       + self.C2 * random.random() * (gbest - x)
                # v[i] = self.W*v[i] + self.C1*random.random()*(pbest[i]-x)\
                       # + self.C2*random.random()*(gbest-x)
                initial_a=x+v[i]
                index_work=initial_a.argsort()
                job=[]
                for j in range(len(jobs)):
                    job.append(jobs[index_work[j]])
                job=np.array(job).reshape(1,len(jobs))
                machine_new,time_new = self.back_MT(job,Ma_1,Tm_1)
                C_finish,_,_,_,_=obj_fjsp.caculate(job,machine_new,time_new)
                #自我判断这里少了一个判断最大完工时间是否小于已知种群中的某一个完工时间
                Wmax=0
                Wmin=0
                if C_finish<answer[i]:
                    res += 1
                    work_job[i]=job
                    job_initial[i]=initial_a
                    answer[i]=C_finish
                    pbest[i]=initial_a
            #判断适应度值的变化率

            W =res/len(jobs)

            # if res > 0:
            #     W = W * 1.1
            # else:
            #     W = 0.9 * W
            #均匀交叉
            si=0
            while si < self.popsize-1 :
                r1 = random.randint(0,self.popsize-1)
                r2 = random.randint(0,self.popsize-1)
                p1 = work_job[r1:r1+1]
                p2 = work_job[r2:r2+1]
                ma,mat,wcr = self.to_MT(p1,work_machine[r1:r1+1],work_time[r1:r1+1])
                c = []
                for j in range(len(jobs)):
                    r = random.randint(1, 2)
                    if r == 1:
                        c.append(p1[0,j])
                    if r == 2:
                        c.append(p2[0,j])
                flag=True
                for i in range(self.job_num):
                    if c.count(i)>jobs.count(i):
                        flag=False
                        break
                if flag:
                    si+=1
                    c=np.array(c).reshape(1,len(jobs))
                    Ma, Mt = self.back_MT(c, ma, mat)
                    C_finish, _, _, _, _ = obj_fjsp.caculate(c,Ma,Mt)
                    if C_finish<answer[r1]:
                        work_job[r1]=c
                        answer[r1]=C_finish



            # 机器编码
            for i in range(0,self.popsize,2):
                job,machine,machine_time=work_job[i:i+1],work_machine[i:i+1],work_time[i:i+1]
                Ma_1,Tm_1,wcross = self.to_MT(job,machine,machine_time)
                job1,machine1,machine_time1=work_job[i+1:i+2],work_machine[i+1:i+2],work_time[i+1:i+2]
                Ma_2,Tm_2,wcross=self.to_MT(job1,machine1,machine_time1)
                Mc1, Mc2, Tc1, Tc2=self.mac_cross(Ma_1,Tm_1,Ma_2,Tm_2,wcross)
                #第一个机器编码
                machine_new1,time_new1=self.back_MT(job,Mc1,Tc1)
                C_finish,_,_,_,_=obj_fjsp.caculate(job,machine_new1,time_new1)
                if(C_finish<answer[i]):
                    work_machine[i]=machine_new1[0]
                    work_time[i]=time_new1[0]
                    answer[i]=C_finish
                # 第二个机器编码
                machine_new2, time_new2 = self.back_MT(job1, Mc2, Tc2)
                C_finish, _, _, _, _ = obj_fjsp.caculate(job, machine_new2, time_new2)
                if (C_finish < answer[i]):
                    work_machine[i+1] = machine_new2[0]
                    work_time[i+1] = time_new2[0]
                    answer[i+1] = C_finish
            best_index=answer.index(min(answer))
            gbest=job_initial[best_index]
            result.append(answer[best_index])
        # print(answer[best_index],file=log)
        return work_job[best_index],work_machine[best_index],work_time[best_index],result
# log = open('log.txt',mode='a',encoding='utf-8')
# # ho = pso([10,15,0.5],200,100,[1,2,2])
# _,job_num,machine_num = get_job_mochine_num.read(1)
# ho = pso([job_num,machine_num,config.machine_pro],config.generation,config.select_num,config.param)
# a,b,c,d=ho.pso_total(1)
# job,machine,machine_time = np.array([a]),np.array([b]),np.array([c])
# obj_fjsp.draw(job,machine,machine_time)
# log.close()