"""by lyuwenyu
"""

import os 
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import argparse

import src.misc.dist as dist
from src.core import YAMLConfig
from src.solver import TASKS


def main(args, ) -> None:
    '''main
    '''
    dist.init_distributed()
    if args.seed is not None:
        dist.set_seed(args.seed)

    assert not all([args.tuning, args.resume]), \
        'Only support from_scrach or resume or tuning at one time'

    cfg = YAMLConfig(
        args.config,
        resume=args.resume, 
        use_amp=args.amp,
        tuning=args.tuning
    )
    # cfg.yaml_cfg['task'] 获取任务类型名称
    # TASKS[cfg.yaml_cfg['task']] 根据任务类型从TASKS字典中获取对应的类
    # 使用cfg作为参数初始化该类实例
    solver = TASKS[cfg.yaml_cfg['task']](cfg)
    # （cfg）是将上面的参数传递给TASKS[cfg.yaml_cfg['task']]的DetSolver类中


    if args.test_only:
        solver.val()
    else:
        solver.fit()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, default=r'F:\LearnDL\RT-DETR-main\RT-DETR-main\rtdetr_pytorch\configs\rtdetr\rtdetr_r18vd_6x_coco.yml')
    parser.add_argument('--resume', '-r', type=str, default=r'F:\LearnDL\RT-DETR-main\RT-DETR-main\rtdetr_pytorch\ResNet18_vd_pretrained_from_paddle.pth')
    parser.add_argument('--tuning', '-t', type=str, )
    parser.add_argument('--test-only', action='store_true', default=False,)
    parser.add_argument('--amp', action='store_true', default=False,)
    parser.add_argument('--seed', type=int, help='seed',)
    args = parser.parse_args()
    print(args)

    main(args)
