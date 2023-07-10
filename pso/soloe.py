import copy
import random

from rbf.RBF import RBFNet,rbf_predict
from utils import get_job_mochine_num
import numpy as np
import config
from FJSP.FJSP import FJSP
from get_data.data_deal import data_deal
from pylab import mpl
from rbf.RBF import Model_Selection
from utils.index_bootstraps import index_bootstrap
from rbf import RBF

mpl.rcParams['font.sans-serif'] = ['SimHei']

class test_pso():
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

    # 找列表中最大值，和第二大值的索引
    def find_max_and_second_max(self,lst):
        max_value = float('-inf')  # 将最大值初始化为负无穷大
        second_max_value = float('-inf')  # 将第二大值初始化为负无穷大
        max_index = len(lst)  # 最大值位置的索引
        second_max_index = len(lst)  # 第二大值位置的索引

        for index, value in enumerate(lst):
            if value > max_value:
                second_max_value = max_value
                second_max_index = max_index
                max_value = value
                max_index = index
            elif value > second_max_value:
                second_max_value = value
                second_max_index = index

        return max_index, second_max_index

    def find_min_and_second_min(self,lst):
        min_value = float('inf')  # 将最大值初始化为无穷大
        second_min_value = float('inf')  # 将第二值初始化为无穷大
        min_index = len(lst)  # 最小值位置的索引
        second_min_index = len(lst)  # 第二值位置的索引

        for index,value in enumerate(lst):
            if value<min_value:
                second_min_value=min_value
                second_min_index=min_index
                min_value=value
                min_index=index
            elif value<second_min_value:
                second_min_value=value
                second_min_index=index
        return  min_index,second_min_index
    # 粒子群算法
    def pso_total(self,iter):
        global obj_fjsp
        obj_datadeal = data_deal(self.job_num,self.machine_num)
        Tmachine, Tmachinetime, process_mac_num, jobs, tom=obj_datadeal.time_mac_job_pro(iter)
        param_data=[Tmachine,Tmachinetime,process_mac_num,jobs,tom]
        obj_fjsp=FJSP(self.job_num,self.machine_num,self.pi,param_data)
        answer,result = [],[]
        job_initial,pbest = np.zeros((self.popsize,len(jobs))),np.zeros((self.popsize,len(jobs)))
        work_job,work_machine,work_time = np.zeros((self.popsize,len(jobs))),np.zeros((self.popsize,len(jobs))),np.zeros((self.popsize,len(jobs)))
        v = np.zeros((self.popsize,len(jobs)))
        W = self.W
        a = work_job.shape[0]
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
                print('第'+str(iter)+'代'+'种群初始的最小最大完工时间: %0.f'%(min(answer)))

            if gen < self.generation:
                # 粒子群算法
                # job_swarm=np.zeros((self.popsize,len(jobs)))
                # machine_swarm =np.zeros((self.popsize,len(jobs)))
                # machinetime_swarm = np.zeros((self.popsize,len(jobs)))
                for i in range(self.popsize):
                    job, machine, machine_time = work_job[i:i + 1], work_machine[i:i + 1], work_time[i:i + 1]
                    C_finish,_,_,_,_ = obj_fjsp.caculate(job,machine,machine_time)
                    answer[i]=C_finish
                # 构建rbf代理模型
                # train_x = np.random.rand(self.popsize, work_job.shape[1] * 2)
                train_x = np.random.rand(self.popsize, work_job.shape[1] * 3)
                for i in range(work_job.shape[0]):
                    train_x[i][0:len(work_job[i])] = work_job[i]
                    train_x[i][len(work_job[i]):2 * len(work_job[i])] = work_machine[i]
                    train_x[i][2 * len(work_job[i]):] = work_time[i]
                center_locals, weight_locals, bias_locals, spread_locals, = [], [], [], []
                # idxs_users = np.random.choice(range(config.model_num), config.model_num,
                #                               replace=False)  # m代表元素数量，range是取值范围
                train_y = np.array(answer).reshape(-1, 1)
                for i in range(config.model_num):
                    data_index = index_bootstrap(config.model_num, config.boot_prob)
                    RBF = RBFNet(k=config.k)
                    local_c, local_w, local_b, local_s = RBF.local_update(train_x[data_index], train_y[data_index])
                    center_locals.append(local_c)
                    weight_locals.append(local_w)
                    bias_locals.append(local_b)
                    spread_locals.append(local_s)
                res=0
                for i in range(self.popsize):
                    job,machine,machine_time=work_job[i:i+1],work_machine[i:i+1],work_time[i:i+1]
                    Ma_1,Tm_1,Wcross = self.to_MT(job,machine,machine_time)
                    x = job_initial[i]
                    v[i] = W*v[i] + self.C1*random.random()*(pbest[i]-x)\
                           + self.C2*random.random()*(gbest-x)
                    initial_a=x+v[i]
                    index_work=initial_a.argsort()
                    job=[]
                    for j in range(len(jobs)):
                        job.append(jobs[index_work[j]])
                    job = np.array(job).reshape(1, len(jobs))
                    machine_new, time_new = self.back_MT(job, Ma_1, Tm_1)
                    C_finish,_,_,_,_=obj_fjsp.caculate(job,machine_new,time_new)
                    if C_finish<answer[i]:
                        res+=1
                        work_job[i]=job[0]
                        work_machine[i]=machine_new[0]
                        work_time[i]=time_new[0]
                        job_initial[i]=initial_a
                        answer[i]=C_finish
                        pbest[i]=initial_a
                W = res / self.popsize


                # 锦标赛选择策略
                S_job,S_machine,S_time = np.zeros((self.popsize,len(jobs))),np.zeros((self.popsize,len(jobs))),np.zeros((self.popsize,len(jobs)))
                S_answer = [0]*len(answer)
                for i in range(self.popsize):
                    select_index = random.sample(range(len(answer)),config.tournament_size)
                    fit = max(answer)
                    for j in select_index:
                        if answer[j] < fit:
                            fit=answer[j]
                            index=j
                    S_job[i]=work_job[index]
                    S_machine[i]=work_machine[index]
                    S_time[i]=work_time[index]
                    S_answer[i]=answer[index]
                work_job=S_job
                work_machine=S_machine
                work_time=S_time
                answer=S_answer
                #pox 交叉
                # 每次对最差的两个个体进行交叉，得到新的个体
                for i in range(self.popsize):
                # for i in range(int(self.popsize/2)):
                    r1 ,r2= self.find_max_and_second_max(answer)   # 找适应度最大的两个个体进行pox交叉
                    p1 = work_job[r1].reshape(1,len(work_job[r1]))
                    p2 = work_job[r2].reshape(1,len(work_job[r2]))  # 两条父染色体
                    seq = [i + 1 for i in range(self.job_num)]
                    random_length1 = np.random.randint(2, len(seq) - 1)
                    for i in range(random_length1):  # 选出需要交叉的工件
                        index = np.random.randint(0, len(seq))
                        seq.pop(index)
                    set1 = set(seq)  # 得到需要交叉的工件的集合
                    child1 = copy.deepcopy(p1)
                    child2 = copy.deepcopy(p2)
                    remain1 = [i for i in range(len(p1[0])) if p1[0,i] in set1]
                    remain2 = [i for i in range(len(p1[0])) if p2[0,i] in set1]
                    cursor1, cursor2 = 0, 0
                    for i in range(len(p1[0])):
                        if p2[0,i] in set1:
                            child1[0,remain1[cursor1]] = p2[0,i]
                            cursor1 += 1
                        if p1[0,i] in set1:
                            child2[0,remain2[cursor2]] = p1[0,i]
                            cursor2 += 1
                    ma1 ,mt1 = work_machine[r1].reshape(1,len(work_job[r1])),work_time[r1].reshape(1,len(work_time[r1]))
                    ma11,mt11,wrc = self.to_MT(p1,ma1,mt1)
                    ma1_new,mt1_new = self.back_MT(child1,ma11,mt11)
                    ma2 ,mt2 = work_machine[r2].reshape(1,len(work_machine[r2])) ,work_time[r2].reshape(1,len(work_time[r2]))
                    ma22,mt22,wrc = self.to_MT(p2,ma2,mt2)
                    ma2_new,mt2_new = self.back_MT(child2,ma22,mt22)
                    # C_finish1,_,_,_,_ = obj_fjsp.caculate(child1,ma1_new,mt1_new)
                    # C_finish2,_,_,_,_ = obj_fjsp.caculate(child2,ma2_new,mt2_new)
                    jm1 = np.random.rand(1, work_job.shape[1] * 3)
                    jm1[0, 0:work_job.shape[1]] = child1[0]
                    jm1[0, work_job.shape[1]:2 * work_job.shape[1]] = ma1_new[0]
                    jm1[0, work_job.shape[1] * 2:] = mt1_new[0]
                    jm2 = np.random.rand(1, work_job.shape[1] * 3)
                    jm2[0, 0:work_job.shape[1]] = child2[0]
                    jm2[0, work_job.shape[1]:2 * work_job.shape[1]] = ma2_new[0]
                    jm2[0, work_job.shape[1] * 2:] = mt2_new[0]
                    # C_finish1 = rbf_predict(center_locals[r1],weight_locals[r1],bias_locals[r1],spread_locals[r1],jm1)
                    # C_finish2 = rbf_predict(center_locals[r2],weight_locals[r2],bias_locals[r2],spread_locals[r2],jm2)
                    C_, _, _, _, _ = obj_fjsp.caculate(child1, ma1_new, mt1_new)
                    if C_ < answer[r1]:
                        C_finish1 = rbf_predict(center_locals[r1], weight_locals[r1], bias_locals[r1],
                                                spread_locals[r1], jm1)
                        if C_finish1 < answer[r1]:
                            answer[r1] = C_finish1
                            work_job[r1] = child1[0]
                            work_machine[r1] = ma1_new[0]
                            work_time[r1] = mt1_new[0]
                    C_, _, _, _, _ = obj_fjsp.caculate(child2, ma2_new, mt2_new)
                    if C_ < answer[r2]:
                        C_finish2 = rbf_predict(center_locals[r2], weight_locals[r2], bias_locals[r2],
                                                spread_locals[r2], jm2)
                        if C_finish2 < answer[r2]:
                            answer[r2]=C_finish2
                            work_job[r2]=child2[0]
                            work_machine[r2]=ma2_new[0]
                            work_time[r2]=mt2_new[0]
                # 每次用最好的两个进行交叉，与最好个体适应度进行比较
                for i in range(int(self.popsize/4)):
                # for i in range(int(self.popsize/2)):
                    r1 ,r2= self.find_min_and_second_min(answer)   # 找适应度最小的两个个体进行pox交叉
                    p1 = work_job[r1].reshape(1,len(work_job[r1]))
                    p2 = work_job[r2].reshape(1,len(work_job[r2]))  # 两条父染色体
                    seq = [i  for i in range(self.job_num)]
                    random_length1 = np.random.randint(2, len(seq) - 1)
                    for i in range(random_length1):  # 选出需要交叉的工件
                        index = np.random.randint(0, len(seq))
                        seq.pop(index)
                    set1 = set(seq)  # 得到需要交叉的工件的集合
                    child1 = copy.deepcopy(p1)
                    child2 = copy.deepcopy(p2)
                    remain1 = [i for i in range(len(p1[0])) if p1[0,i] in set1]
                    remain2 = [i for i in range(len(p1[0])) if p2[0,i] in set1]
                    cursor1, cursor2 = 0, 0
                    for i in range(len(p1[0])):
                        if p2[0,i] in set1:
                            child1[0,remain1[cursor1]] = p2[0,i]
                            cursor1 += 1
                        if p1[0,i] in set1:
                            child2[0,remain2[cursor2]] = p1[0,i]
                            cursor2 += 1
                    ma1 ,mt1 = work_machine[r1].reshape(1,len(work_job[r1])),work_time[r1].reshape(1,len(work_time[r1]))
                    ma11,mt11,wrc = self.to_MT(p1,ma1,mt1)
                    ma1_new,mt1_new = self.back_MT(child1,ma11,mt11)
                    ma2 ,mt2 = work_machine[r2].reshape(1,len(work_machine[r2])) ,work_time[r2].reshape(1,len(work_time[r2]))
                    ma22,mt22,wrc = self.to_MT(p2,ma2,mt2)
                    ma2_new,mt2_new = self.back_MT(child2,ma22,mt22)
                    C_finish1,_,_,_,_ = obj_fjsp.caculate(child1,ma1_new,mt1_new)
                    C_finish2,_,_,_,_ = obj_fjsp.caculate(child2,ma2_new,mt2_new)
                    # if  C_finish1 < C_finish2 :
                    #     child = child1
                    #     C_ = C_finish1
                    #     ma,mt=ma1_new,mt1_new
                    # else:
                    #     child = child2
                    #     C_ = C_finish2
                    #     ma,mt=ma2_new,mt2_new
                    if C_finish1 < answer[r1]:
                        answer[r1] = C_finish1
                        work_job[r1] = child1[0]
                        work_machine[r1] = ma1_new[0]
                        work_time[r1] = mt1_new[0]
                    if C_finish2 < answer[r2]:
                        answer[r2]=C_finish2
                        work_job[r2]=child2[0]
                        work_machine[r2]=ma2_new[0]
                        work_time[r2]=mt2_new[0]

                # 机器编码

                for i in range(0,self.popsize):
                    r1, r2 = self.find_max_and_second_max(answer)  # 找适应度最大的两个个体进行pox交叉
                    job1, machine, machine_time = work_job[r1].reshape(1, len(work_job[r1])),work_machine[r1].reshape(1, len(work_machine[r1])),work_time[r1].reshape(1, len(work_time[r1]))
                    job2,machine2,machine_time2 = work_job[r2].reshape(1, len(work_job[r2])),work_machine[r2].reshape(1, len(work_machine[r2])),work_time[r2].reshape(1, len(work_time[r2]))  # 两条父染色体
                    # job1,machine,machine_time=work_job[i:i+1],work_machine[i:i+1],work_time[i:i+1]
                    Ma_1,Tm_1,wcross = self.to_MT(job1,machine,machine_time)
                    # job2,machine2,machine_time2=work_job[i+1:i+2],work_machine[i+1:i+2],work_time[i+1:i+2]
                    Ma_2,Tm_2,wcross=self.to_MT(job2,machine2,machine_time2)
                    Mc1, Mc2, Tc1, Tc2=self.mac_cross(Ma_1,Tm_1,Ma_2,Tm_2,wcross)

                    # rbf预测
                    machine_new1, time_new1 = self.back_MT(job1, Mc1, Tc1)
                    jm1 = np.random.rand(1, work_job.shape[1] * 3)
                    jm1[0, 0:work_job.shape[1]] = job1[0]
                    jm1[0, work_job.shape[1]:2 * work_job.shape[1]] = machine_new1[0]
                    jm1[0, work_job.shape[1] * 2:] = time_new1[0]
                    C_finish1 = rbf_predict(center_locals[r1], weight_locals[r1], bias_locals[r1],
                                                        spread_locals[r1], jm1)
                    machine_new2, time_new2 = self.back_MT(job2, Mc2, Tc2)
                    jm2 = np.random.rand(1, work_job.shape[1] * 3)
                    jm2[0, 0:work_job.shape[1]] = job2[0]
                    jm2[0, work_job.shape[1]:2 * work_job.shape[1]] = machine_new2[0]
                    jm2[0, work_job.shape[1] * 2:] = time_new2[0]

                    C_finish2 = rbf_predict(center_locals[r2], weight_locals[r2], bias_locals[r2],
                                                        spread_locals[r2], jm2)
                    if C_finish1 < answer[r1]:
                        work_machine[r1] = machine_new1[0]
                        work_time[r1] = time_new1[0]
                        answer[r1] = C_finish1

                    if C_finish2 < answer[r2]:
                        work_machine[r2] = machine_new2[0]
                        work_time[r2] = time_new2[0]
                        answer[r2] = C_finish2
                    #第一个机器编码
                    # machine_new1,time_new1=self.back_MT(job1,Mc1,Tc1)
                    # C_finish, _, _, _, _ = obj_fjsp.caculate(job1, machine_new1, time_new1)
                    # if (C_finish1 < answer[r1]):
                    #     work_machine[i] = machine_new1[0]
                    #     work_time[i] = time_new1[0]
                    #     answer[i] = C_finish1
                    # 第二个机器编码
                    # machine_new2, time_new2 = self.back_MT(job2, Mc2, Tc2)
                    # C_finish, _, _, _, _ = obj_fjsp.caculate(job2, machine_new2, time_new2)
                    # if (C_finish2 < answer[r2]):
                    #     work_machine[i + 1] = machine_new2[0]
                    #     work_time[i + 1] = time_new2[0]
                    #     answer[i + 1] = C_finish2
                    # 第二个机器编码


                best_index=answer.index(min(answer))
                gbest=job_initial[best_index]
                result.append(answer[best_index])
                rbf_model = [center_locals, weight_locals, bias_locals, spread_locals]
            # if gen == self.generation-1:
            #     # 粒子群算法
            #     for i in range(self.popsize):
            #         job, machine, machine_time = work_job[i:i + 1], work_machine[i:i + 1], work_time[i:i + 1]
            #         Ma_1, Tm_1, Wcross = self.to_MT(job, machine, machine_time)
            #         x = job_initial[i]
            #         v[i] = W * v[i] + self.C1 * random.random() * (pbest[i] - x) \
            #                + self.C2 * random.random() * (gbest - x)
            #         initial_a = x + v[i]
            #         index_work = initial_a.argsort()
            #         job = []
            #         for j in range(len(jobs)):
            #             job.append(jobs[index_work[j]])
            #         job = np.array(job).reshape(1, len(jobs))
            #         machine_new, time_new = self.back_MT(job, Ma_1, Tm_1)
            #         C_finish,_,_,_,_=obj_fjsp.caculate(job,machine_new,time_new)
            #         if C_finish<answer[i]:
            #             work_job[i]=job[0]
            #             work_machine[i]=machine_new[0]
            #             work_time[i]=time_new[0]
            #             job_initial[i]=initial_a
            #             answer[i]=C_finish
            #             pbest[i]=initial_a
            #
            #     for i in range(0, self.popsize, 2):
            #         job1, machine1, machine_time1 = work_job[i:i + 1], work_machine[i:i + 1], work_time[i:i + 1]
            #         Ma_1, Tm_1, wcross = self.to_MT(job1, machine1, machine_time1)
            #         job2, machine2, machine_time2 = work_job[i + 1:i + 2], work_machine[i + 1:i + 2], work_time[i + 1:i + 2]
            #         Ma_2, Tm_2, wcross = self.to_MT(job2, machine2, machine_time2)
            #         Mc1, Mc2, Tc1, Tc2 = self.mac_cross(Ma_1, Tm_1, Ma_2, Tm_2, wcross)
            #         # 第一个机器编码
            #         machine_new1, time_new1 = self.back_MT(job1, Mc1, Tc1)
            #         C_finish, _, _, _, _ = obj_fjsp.caculate(job1, machine_new1, time_new1)
            #         if (C_finish < answer[i]):
            #             work_machine[i] = machine_new1[0]
            #             work_time[i] = time_new1[0]
            #             answer[i] = C_finish
            #         # 第二个机器编码
            #         machine_new2, time_new2 = self.back_MT(job2, Mc2, Tc2)
            #         C_finish, _, _, _, _ = obj_fjsp.caculate(job2, machine_new2, time_new2)
            #         if (C_finish < answer[i+1]):
            #             work_machine[i + 1] = machine_new2[0]
            #             work_time[i + 1] = time_new2[0]
            #             answer[i + 1] = C_finish
            #     best_index = answer.index(min(answer))
            #     gbest = job_initial[best_index]
            #     result.append(answer[best_index])
            #     # train_x = np.random.rand(self.popsize, work_job.shape[1] * 2)
            #     # for i in range(work_job.shape[0]):
            #     #     train_x[i][0:len(work_job[i])] = work_job[i]
            #     #     train_x[i][len(work_job[i]):] = work_machine[i]
            #     # center_locals, weight_locals, bias_locals, spread_locals, = [], [], [], []
            #     # idxs_users = np.random.choice(range(config.model_num), config.model_num,
            #     #                               replace=False)  # m代表元素数量，range是取值范围
            #     # train_y = np.array(answer).reshape(-1, 1)
            #     # for idx in idxs_users:
            #     #     data_index = index_bootstrap(config.model_num, config.boot_prob)
            #     #     RBF = RBFNet(k=config.k)
            #     #     local_c, local_w, local_b, local_s = RBF.local_update(train_x[data_index], train_y[data_index])
            #     #     center_locals.append(local_c)
            #     #     weight_locals.append(local_w)
            #     #     bias_locals.append(local_b)
            #     #     spread_locals.append(local_s)
                # rbf_model = [center_locals,weight_locals,bias_locals,spread_locals]
        return work_job[best_index],work_machine[best_index],work_time[best_index],result,rbf_model
