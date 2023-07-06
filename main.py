import time

import numpy as np
import config
from FJSP.FJSP import FJSP
from get_data.data_deal import data_deal
from pso.pso import pso
from pso.test_pso import test_pso
from pso import addSelect_pso
from utils import get_job_mochine_num
from utils.index_bootstraps import index_bootstrap
from rbf.RBF import RBFNet
import warnings
warnings.filterwarnings("ignore")

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    t0=time.time()
    # log = open('log.rbf.txt', mode='a', encoding='utf-8')
    # _, job_num, machine_num = get_job_mochine_num.read(1)
    # obj_datadeal = data_deal(job_num, machine_num)
    # Tmachine, Tmachinetime, process_mac_num, jobs, tom = obj_datadeal.time_mac_job_pro(1)
    # param_data = [Tmachine, Tmachinetime, process_mac_num, jobs, tom]
    #
    # ho = test_pso([job_num, machine_num, config.machine_pro], config.generation, config.popsize, config.param)
    # # ho = pso([job_num, machine_num, config.machine_pro], config.generation, config.popsize, config.param)
    # a, b, c, d = ho.pso_total(1)
    # job, machine, machine_time = np.array([a]), np.array([b]), np.array([c])
    # obj_fjsp = FJSP(job_num, machine_num, config.machine_pro, param_data)
    # res, _, _, _, _ = obj_fjsp.caculate(job, machine, machine_time)
    # print('MK0' + str(1) + '.fjs: ', res, file=log)
    # print('该算例的工序集合：', job, file=log)
    # print('该算例的机器集合：', machine, file=log)
    # print('该算例的加工时间集合：', machine_time, file=log)
    # obj_fjsp.draw(job, machine, machine_time)
    log = open('logrbf.txt', mode='a', encoding='utf-8')
    for i in range(1,10):
        _, job_num, machine_num = get_job_mochine_num.read(i)
        obj_datadeal = data_deal(job_num, machine_num)
        Tmachine, Tmachinetime, process_mac_num, jobs, tom = obj_datadeal.time_mac_job_pro(i)
        param_data = [Tmachine, Tmachinetime, process_mac_num, jobs, tom]

        # ho = test_pso([job_num, machine_num, config.machine_pro], config.generation, config.popsize, config.param)
        # ho = pso([job_num, machine_num, config.machine_pro], config.generation, config.popsize, config.param)
        ho = addSelect_pso.test_pso([job_num, machine_num, config.machine_pro], config.generation, config.popsize, config.param)
        a, b, c, d, rbf_model = ho.pso_total(i)
        job, machine, machine_time = np.array([a]), np.array([b]), np.array([c])
        obj_fjsp=FJSP(job_num,machine_num,config.machine_pro,param_data)
        res,_,_,_,_=obj_fjsp.caculate(job,machine,machine_time)
        print('MK0'+str(i)+'.fjs: ',res,file=log)
        # print('该算例的工序集合：',job,file=log)
        # print('该算例的机器集合：',machine,file=log)
        # print('该算例的加工时间集合：', machine_time, file=log)
        obj_fjsp.draw(job, machine, machine_time,i)
    log.close()
    t1 = time.time()
    print(t1-t0)

