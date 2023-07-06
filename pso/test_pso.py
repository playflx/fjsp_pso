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

            if gen < self.generation-1:
                # 粒子群算法
                job_swarm=np.zeros((self.popsize,len(jobs)))
                machine_swarm =np.zeros((self.popsize,len(jobs)))
                machinetime_swarm = np.zeros((self.popsize,len(jobs)))
                for i in range(self.popsize):
                    job, machine, machine_time = work_job[i:i + 1], work_machine[i:i + 1], work_time[i:i + 1]
                    C_finish,_,_,_,_ = obj_fjsp.caculate(job,machine,machine_time)
                    answer[i]=C_finish
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
                        work_job[i]=job
                        job_initial[i]=initial_a
                        answer[i]=C_finish
                        pbest[i]=initial_a

                train_x = np.random.rand(self.popsize, work_job.shape[1] * 2)
                train_x = np.random.rand(self.popsize, work_job.shape[1] * 3)
                for i in range(work_job.shape[0]):
                    train_x[i][0:len(work_job[i])] = work_job[i]
                    train_x[i][len(work_job[i]):2*len(work_job[i])] = work_machine[i]
                    train_x[i][2*len(work_job[i]):] = work_time[i]
                center_locals, weight_locals, bias_locals, spread_locals, = [], [], [], []
                idxs_users = np.random.choice(range(config.model_num), config.model_num,
                                              replace=False)  # m代表元素数量，range是取值范围
                train_y = np.array(answer).reshape(-1, 1)
                for idx in idxs_users:
                    data_index = index_bootstrap(config.model_num, config.boot_prob)
                    RBF = RBFNet(k=config.k)
                    local_c, local_w, local_b, local_s = RBF.local_update(train_x[data_index], train_y[data_index])
                    center_locals.append(local_c)
                    weight_locals.append(local_w)
                    bias_locals.append(local_b)
                    spread_locals.append(local_s)
                W = res/self.popsize
                si = 0
                while (si < self.popsize - 1):
                    r1 = random.randint(0, self.popsize - 1)
                    r2 = random.randint(0, self.popsize - 1)
                    p1 = work_job[r1:r1 + 1]
                    p2 = work_job[r2:r2 + 1]
                    ma, mat, wcr = self.to_MT(p1, work_machine[r1:r1 + 1], work_time[r1:r1 + 1])
                    c = []
                    for j in range(len(jobs)):
                        r = random.randint(1, 2)
                        if r == 1:
                            c.append(p1[0, j])
                        if r == 2:
                            c.append(p2[0, j])
                    flag = True
                    for i in range(self.job_num):
                        if c.count(i) > jobs.count(i):
                            flag = False
                            break
                    if flag:

                        c = np.array(c).reshape(1, len(jobs))
                        Ma, Mt = self.back_MT(c, ma, mat)
                        # jm = np.random.rand(1, work_job.shape[1] * 2)
                        jm = np.random.rand(1, work_job.shape[1] * 3)
                        jm[0,0:work_job.shape[1]]=c[0]
                        jm[0,work_job.shape[1]:2*work_job.shape[1]] = Ma[0]
                        jm[0,work_job.shape[1]*2:] = Mt[0]
                        ans = rbf_predict(center_locals[si], weight_locals[si], bias_locals[si], spread_locals[si],jm)
                        ans1 = obj_fjsp.caculate(c,Ma,Mt)
                        if ans < answer[si] :
                            work_job[si] = c
                            answer[si] = ans[0][0]
                        # C_finish, _, _, _, _ = obj_fjsp.caculate(c, Ma, Mt)
                        # if C_finish < answer[si]:
                        #     work_job[si] = c
                        #     answer[si] = C_finish
                        si += 1
                # 机器编码
                job_1 = np.zeros((self.popsize, len(jobs)))
                machine_1 = np.zeros((self.popsize, len(jobs)))
                machinetime_1 = np.zeros((self.popsize, len(jobs)))
                for i in range(0,self.popsize,2):
                    job,machine,machine_time=work_job[i:i+1],work_machine[i:i+1],work_time[i:i+1]
                    Ma_1,Tm_1,wcross = self.to_MT(job,machine,machine_time)
                    job1,machine1,machine_time1=work_job[i+1:i+2],work_machine[i+1:i+2],work_time[i+1:i+2]
                    Ma_2,Tm_2,wcross=self.to_MT(job1,machine1,machine_time1)
                    Mc1, Mc2, Tc1, Tc2=self.mac_cross(Ma_1,Tm_1,Ma_2,Tm_2,wcross)

                    #第一个机器编码
                    machine_new,time_new=self.back_MT(job,Mc1,Tc1)
                    job_1[i]=job
                    machine_1[i]=machine_new
                    machinetime_1[i]=time_new
                    # 第二个机器编码
                    machine_new1, time_new1 = self.back_MT(job1, Mc2, Tc2)
                    job_1[i+1] = job1
                    machine_1[i+1] = machine_new1
                    machinetime_1[i+1] = time_new1
                tmp = np.zeros((work_job.shape[0], config.select_num))
                # pop_1 = np.random.rand(self.popsize, job_1.shape[1] * 2)
                pop_1 = np.random.rand(self.popsize, job_1.shape[1] * 3)
                # bestpop1 = np.random.rand(self.popsize, work_job.shape[1] * 2)
                bestpop1 = np.random.rand(self.popsize, work_job.shape[1] * 3)
                for i in range(work_job.shape[0]):
                    pop_1[i][0:len(job_1[i])] = job_1[i]
                    pop_1[i][len(job_1[i]):2*len(job_1[i])] = machine_1[i]
                    pop_1[i][2*len(job_1[i]):] = machinetime_1[i]
                    bestpop1[i][0:len(work_job[i])] = work_job[i]
                    bestpop1[i][len(work_job[i]):2*len(work_job[i])] = work_machine[i]
                    bestpop1[i][2*len(work_job[i]):] = work_time[i]
                # new_pop_1 = np.array(pop_1).reshape(self.popsize, 2 * job_1.shape[1])
                new_pop_1 = np.array(pop_1).reshape(self.popsize, 3 * job_1.shape[1])
                # bestnew_pop_1 = np.array(bestpop1).reshape(self.popsize, 2 * work_job .shape[1])
                bestnew_pop_1 = np.array(bestpop1).reshape(self.popsize, 3 * work_job.shape[1])
                bestindex = np.argmin(answer)
                # best_pop_1 = bestnew_pop_1[bestindex, :].reshape(-1, 2 * job_1.shape[1])
                best_pop_1 = bestnew_pop_1[bestindex, :].reshape(-1, 3 * job_1.shape[1])
                model_index = Model_Selection(center_locals, weight_locals, bias_locals, spread_locals, best_pop_1,
                                              num_set=config.model_num)
                for i in range(config.select_num):
                    _ = model_index[i]
                    tmp[:,i] = rbf_predict(center_locals[_], weight_locals[_], bias_locals[_], spread_locals[_],
                                    new_pop_1).flatten()
                    for j,tempm in enumerate(tmp[:,i]):
                        if tempm < answer[j] and tempm>0:
                            work_machine[j] = machine_1[j]
                            work_time[j] = machinetime_1[j]
                            answer[j]=tempm
                best_index=answer.index(min(answer))
                gbest=job_initial[best_index]
                result.append(answer[best_index])
            if gen == self.generation-1:
                # 粒子群算法
                for i in range(self.popsize):
                    job, machine, machine_time = work_job[i:i + 1], work_machine[i:i + 1], work_time[i:i + 1]
                    Ma_1, Tm_1, Wcross = self.to_MT(job, machine, machine_time)
                    x = job_initial[i]
                    v[i] = W * v[i] + self.C1 * random.random() * (pbest[i] - x) \
                           + self.C2 * random.random() * (gbest - x)
                    initial_a = x + v[i]
                    index_work = initial_a.argsort()
                    job = []
                    for j in range(len(jobs)):
                        job.append(jobs[index_work[j]])
                    job = np.array(job).reshape(1, len(jobs))
                    machine_new, time_new = self.back_MT(job, Ma_1, Tm_1)
                    C_finish,_,_,_,_=obj_fjsp.caculate(job,machine_new,time_new)
                    if C_finish<answer[i]:
                        work_job[i]=job
                        job_initial[i]=initial_a
                        answer[i]=C_finish
                        pbest[i]=initial_a

                for i in range(0, self.popsize, 2):
                    job1, machine1, machine_time1 = work_job[i:i + 1], work_machine[i:i + 1], work_time[i:i + 1]
                    Ma_1, Tm_1, wcross = self.to_MT(job1, machine1, machine_time1)
                    job2, machine2, machine_time2 = work_job[i + 1:i + 2], work_machine[i + 1:i + 2], work_time[i + 1:i + 2]
                    Ma_2, Tm_2, wcross = self.to_MT(job2, machine2, machine_time2)
                    Mc1, Mc2, Tc1, Tc2 = self.mac_cross(Ma_1, Tm_1, Ma_2, Tm_2, wcross)
                    # 第一个机器编码
                    machine_new1, time_new1 = self.back_MT(job1, Mc1, Tc1)
                    C_finish, _, _, _, _ = obj_fjsp.caculate(job1, machine_new1, time_new1)
                    if (C_finish < answer[i]):
                        work_machine[i] = machine_new1[0]
                        work_time[i] = time_new1[0]
                        answer[i] = C_finish
                    # 第二个机器编码
                    machine_new2, time_new2 = self.back_MT(job2, Mc2, Tc2)
                    C_finish, _, _, _, _ = obj_fjsp.caculate(job2, machine_new2, time_new2)
                    if (C_finish < answer[i+1]):
                        work_machine[i + 1] = machine_new2[0]
                        work_time[i + 1] = time_new2[0]
                        answer[i + 1] = C_finish
                best_index = answer.index(min(answer))
                gbest = job_initial[best_index]
                result.append(answer[best_index])
                train_x = np.random.rand(self.popsize, work_job.shape[1] * 2)
                for i in range(work_job.shape[0]):
                    train_x[i][0:len(work_job[i])] = work_job[i]
                    train_x[i][len(work_job[i]):] = work_machine[i]
                center_locals, weight_locals, bias_locals, spread_locals, = [], [], [], []
                idxs_users = np.random.choice(range(config.model_num), config.model_num,
                                              replace=False)  # m代表元素数量，range是取值范围
                train_y = np.array(answer).reshape(-1, 1)
                for idx in idxs_users:
                    data_index = index_bootstrap(config.model_num, config.boot_prob)
                    RBF = RBFNet(k=config.k)
                    local_c, local_w, local_b, local_s = RBF.local_update(train_x[data_index], train_y[data_index])
                    center_locals.append(local_c)
                    weight_locals.append(local_w)
                    bias_locals.append(local_b)
                    spread_locals.append(local_s)
                rbf_model = [center_locals,weight_locals,bias_locals,spread_locals]
        return work_job[best_index],work_machine[best_index],work_time[best_index],result,rbf_model
# log = open('log.txt',mode='a',encoding='utf-8')
# # ho = pso([10,15,0.5],200,100,[1,2,2])
# _,job_num,machine_num = get_job_mochine_num.read()
# ho = pso([job_num,machine_num,config.machine_pro],config.generation,config.select_num,config.param)
# a,b,c,d=ho.pso_total()
# job,machine,machine_time = np.array([a]),np.array([b]),np.array([c])
# obj_fjsp.draw(job,machine,machine_time)
# log.close()
  #     job=[]
                #     for j in range(len(jobs)):
                #         job.append(jobs[index_work[j]])
                #     job_swarm[i]= np.array(job).reshape(1,len(jobs))
                #     machine_new,time_new = self.back_MT(job_swarm,Ma_1,Tm_1)
                #     machine_swarm[i] = machine_new
                #     machinetime_swarm[i] = time_new
                #     job_initial[i]=initial_a
                #     pbest[i]=initial_a
                # # 均匀交叉
                # # rbf模型预测
                # txp = np.zeros((job_swarm.shape[0],config.select_num))
                # pop = np.random.rand(self.popsize,job_swarm.shape[1], 2)
                # bestpop = np.random.rand(self.popsize, job_swarm.shape[1], 2)
                # for i in range(work_job.shape[0]):
                #     pop[i][:, 0] = job_swarm[i]
                #     pop[i][:, 1] = machine_swarm[i]
                #     # pop[i][:, 2] = machinetime_swarm[i]
                #     bestpop[i][:,0] = work_job[i]
                #     bestpop[i][:,1] = work_machine[i]
                #     # bestpop[i][:,2] = work_time[i]
                # new_pop = np.array(pop).reshape(self.popsize, 2 * job_swarm.shape[1])
                # bestnew_pop = np.array(bestpop).reshape(self.popsize, 2 * job_swarm.shape[1])
                # bestindex=np.argmin(answer)
                # best_pop = bestnew_pop[bestindex, :].reshape(-1,3*job_swarm.shape[1])
                # model_index = Model_Selection(center_locals, weight_locals, bias_locals, spread_locals, best_pop,
                #                               num_set=config.model_num)
                # for i in range(config.select_num):
                #     _ = model_index[i]
                #     txp[:,i] = rbf_predict(center_locals[_], weight_locals[_], bias_locals[_], spread_locals[_],
                #                     new_pop).flatten()
                #     for k,temp in enumerate(txp[:,i]):
                #         if temp < max(answer) and temp>0:
                #             index_answer = np.argmax(answer)
                #             work_job[index_answer]=pop[k][:,0]
                #             answer[index_answer]=temp
# 构建rbf代理模型
            # train_x = np.random.rand(self.popsize,work_job.shape[1],3)
            # for i in range(work_job.shape[0]):
            #     train_x[i][:,0] = work_job[i]
            #     train_x[i][:,1] = work_machine[i]
            #     train_x[i][:,2] = work_time[i]
            # train_x = np.random.rand(self.popsize, work_job.shape[1] * 2)
            # for i in range(work_job.shape[0]):
            #     a=work_job[i]
            #     train_x[i][0:len(work_job[i])] = work_job[i]
            #     train_x[i][len(work_job[i]):] = work_machine[i]
            #     # train_x[i][:,2] = work_time[i]
            # train_x = np.array(train_x).reshape(self.popsize, 2 * work_job.shape[1])
            # center_locals, weight_locals, bias_locals, spread_locals, = [], [], [], []
            # idxs_users = np.random.choice(range(config.model_num), config.model_num,
            #                               replace=False)  # m代表元素数量，range是取值范围
            # train_y=np.array(answer).reshape(-1,1)
            # for idx in idxs_users:
            #     data_index = index_bootstrap(config.model_num, config.boot_prob)
            #     RBF = RBFNet(k=config.k)
            #     # local_c, local_w, local_b, local_s = RBF.local_update(train_x[data_index], answer[data_index])
            #     local_c, local_w, local_b, local_s = RBF.local_update(train_x[data_index], train_y[data_index])
            #     center_locals.append(local_c)
            #     weight_locals.append(local_w)
            #     bias_locals.append(local_b)
            #     spread_locals.append(local_s)