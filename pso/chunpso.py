import copy
import random
from utils import get_job_mochine_num
import numpy as np
import config
from FJSP.FJSP import FJSP
from get_data.data_deal import data_deal
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']

class pso():
    def __init__(self,param_fjsp,generation,popsize,param_pso,cross_prob,parm_data,mutation_prob):
        self.job_num = param_fjsp[0]  # 工件数
        self.machine_num = param_fjsp[1]  # 机器数
        self.pi = param_fjsp[2]  # 选择机器概率
        self.generation = generation  # 迭代次数
        self.popsize = popsize  # 粒子个数
        self.cross_prob = cross_prob  # 交叉概率
        self.mutation_prob = mutation_prob  # 变异概率
        self.W = param_pso[0]
        self.C1 = param_pso[1]
        self.C2 = param_pso[2]
        self.T_machine, self.T_machinetime, self.process_mac_num, self.work, self.tom = parm_data[0], parm_data[1], \
                                                                                        parm_data[2], parm_data[3], \
                                                                                        parm_data[4]

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
                print('第{}个种群初始的最小最大完工时间:'.format(index),(min(answer)))
            # 锦标赛选择策略
            S_job, S_machine, S_time = np.zeros((self.popsize, len(jobs))), np.zeros(
                (self.popsize, len(jobs))), np.zeros((self.popsize, len(jobs)))
            S_answer = [0] * len(answer)
            for i in range(self.popsize):
                select_index = random.sample(range(len(answer)), config.tournament_size)
                fit = max(answer)
                for j in select_index:
                    if answer[j] < fit:
                        fit = answer[j]
                        index = j
                S_job[i] = work_job[index]
                S_machine[i] = work_machine[index]
                S_time[i] = work_time[index]
                S_answer[i] = answer[index]
            work_job = S_job
            work_machine = S_machine
            work_time = S_time
            answer = S_answer
            #工序粒子
            res = 0
            for i in range(self.popsize):
                if np.random.random() < self.cross_prob:
                    job,machine,machine_time=work_job[i:i+1],work_machine[i:i+1],work_time[i:i+1]
                    Ma_1,Tm_1,Wcross = self.to_MT(job,machine,machine_time)
                    x = job_initial[i]
                    v[i] = W * v[i] + self.C1 * random.random() * (pbest[i] - x) \
                           + self.C2 * random.random() * (gbest - x)
                    initial_a=x+v[i]
                    index_work=initial_a.argsort()
                    job=[]
                    for j in range(len(jobs)):
                        job.append(jobs[index_work[j]])
                    job=np.array(job).reshape(1,len(jobs))
                    machine_new,time_new = self.back_MT(job,Ma_1,Tm_1)
                    C_finish,_,_,_,_=obj_fjsp.caculate(job,machine_new,time_new)

                    if C_finish<answer[i]:
                        res += 1
                        work_job[i]=job
                        work_machine[i]=machine_new
                        work_time[i]=time_new
                        job_initial[i]=initial_a
                        answer[i]=C_finish
                        pbest[i]=initial_a
            #判断适应度值的变化率
            W =res/len(jobs) * W
            #工件pox交叉
            for j in range(0,self.popsize,2):
                if np.random.random() < self.cross_prob:
                    r1 = random.randint(0, self.popsize - 1)
                    r2 = random.randint(0, self.popsize - 1)
                    p1 = work_job[r1:r1+1]
                    p2 = work_job[r2:r2+1]
                    # p1 = work_job[j:j + 1]
                    # p2 = work_job[j+1:j + 2]  # 两条父染色体
                    seq = [i for i in range(self.job_num)]
                    random_length1 = np.random.randint(2, len(seq) - 1)
                    for i in range(random_length1):  # 选出需要交叉的工件
                        index = np.random.randint(0, len(seq))
                        seq.pop(index)
                    set1 = set(seq)  # 得到需要交叉的工件的集合
                    child1 = copy.deepcopy(p1)
                    child2 = copy.deepcopy(p2)
                    remain1 = [i for i in range(len(p1[0])) if p1[0, i] in set1]
                    remain2 = [i for i in range(len(p1[0])) if p2[0, i] in set1]
                    cursor1, cursor2 = 0, 0
                    for i in range(len(p1[0])):
                        if p2[0, i] in set1:
                            child1[0, remain1[cursor1]] = p2[0, i]
                            cursor1 += 1
                        if p1[0, i] in set1:
                            child2[0, remain2[cursor2]] = p1[0, i]
                            cursor2 += 1
                    ma1, mt1 = work_machine[r1:r1 + 1], work_time[r1:r1+1]
                    ma11, mt11, wrc = self.to_MT(p1, ma1, mt1)
                    ma1_new, mt1_new = self.back_MT(child1, ma11, mt11)
                    ma2, mt2 = work_machine[r2:r2+1], work_time[r2:r2+1]
                    ma22, mt22, wrc = self.to_MT(p2, ma2, mt2)
                    ma2_new, mt2_new = self.back_MT(child2, ma22, mt22)
                    C_finish1, _, _, _, _ = obj_fjsp.caculate(child1, ma1_new, mt1_new)
                    C_finish2, _, _, _, _ = obj_fjsp.caculate(child2, ma2_new, mt2_new)
                    if C_finish1 < answer[r1]:
                        answer[r1] = C_finish1
                        work_job[r1]=child1[0]
                        work_machine[r1]=ma1_new[0]
                        work_time[r1]=mt1_new[0]

                    if C_finish2 < answer[r2]:
                        answer[r2] = C_finish2
                        work_job[r2] = child2[0]
                        work_machine[r2],work_time[r2] = ma2_new[0], mt2_new[0]


                    # if C_finish1 < C_finish2:
                    #     child = child1
                    #     C_ = C_finish1
                    #     ma, mt = ma1_new, mt1_new
                    # else:
                    #     child = child2
                    #     C_ = C_finish2
                    #     ma, mt = ma2_new, mt2_new
                    # # if C_ < answer[j]:
                    # #     answer[j] = C_
                    # #     work_job[j] = child[0]
                    # #     work_machine[j] = ma[0]
                    # #     work_time[j] = mt[0]
                    # if C_ < max(answer):
                    #     max_index = np.argmax(answer)
                    #     answer[max_index] = C_
                    #     work_job[max_index] = child[0]
                    #     work_machine[max_index] = ma[0]
                    #     work_time[max_index] = mt[0]


            # 机器均匀交叉操作
            for i in range(0,self.popsize,2):
                if np.random.random() < self.cross_prob:
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
            # 机器变异操作
            for i in range(0,self.popsize):
                if np.random.random() < self.mutation_prob:
                    j = work_job[i].reshape(1,len(work_job[i]))
                    ma = work_machine[i].reshape(1, len(work_machine[i]))
                    mt = work_time[i].reshape(1,len(work_time[i]))
                    Ma, Mt, wcr = self.to_MT(j,ma,mt)
                    for j in range(self.job_num):
                        r = np.random.randint(0,len(Ma[j]))  # 随机选择变异位置
                        index_machine = self.process_mac_num[j][r]  # 得到该工件加工到第几个工序可以使用的机器数
                        index_tom = self.tom[j][r]  # 该工件累计工序数
                        high = index_tom
                        low = index_tom - index_machine
                        _time = self.T_machinetime[j, low:high]
                        _machine = self.T_machine[j, low:high]
                        Mt[j][r] = min(_time)
                        ind = np.argwhere(_time==Mt[j][r])
                        Ma[j][r] = _machine[ind[0,0]]
            best_index=answer.index(min(answer))
            gbest=job_initial[best_index]
            result.append(answer[best_index])
        # print(answer[best_index],file=log)
        return work_job[best_index],work_machine[best_index],work_time[best_index],result
