"""by lyuwenyu
"""

from pprint import pprint
import torch 
import torch.nn as nn 
from torch.utils.data import Dataset, DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.cuda.amp.grad_scaler import GradScaler

from typing import Callable, List, Dict

# 定义特殊变量__all__，显式指定当前模块对外公开的接口，表示仅允许BaseConfig被import *导入
__all__ = ['BaseConfig', ]



class BaseConfig(object):
    # TODO property


    def __init__(self) -> None:
        super().__init__()

        # 任务类型，例如 'detection', 'segmentation'
        self.task :str = None 
        
        # 模型实例
        self._model :nn.Module = None 
        # 后处理模块实例
        self._postprocessor :nn.Module = None 
        # 损失函数实例
        self._criterion :nn.Module = None 
        # 优化器实例
        self._optimizer :Optimizer = None 
        # 学习率调度器实例
        self._lr_scheduler :LRScheduler = None 
        # 训练数据加载器实例
        self._train_dataloader :DataLoader = None 
        # 验证数据加载器实例
        self._val_dataloader :DataLoader = None 
        # EMA模型实例
        self._ema :nn.Module = None 
        # 混合精度训练的梯度缩放器实例
        self._scaler :GradScaler = None 

        # 训练数据集实例
        self.train_dataset :Dataset = None
        # 验证数据集实例
        self.val_dataset :Dataset = None
        # 数据加载器使用的线程数
        self.num_workers :int = 0
        # 数据加载器的collate_fn函数
        self.collate_fn :Callable = None

        # 批量大小
        self.batch_size :int = None
        # 训练时的批量大小
        self._train_batch_size :int = None
        # 验证时的批量大小
        self._val_batch_size :int = None
        # 训练数据是否打乱
        self._train_shuffle: bool = None  
        # 验证数据是否打乱
        self._val_shuffle: bool = None 

        # 评估函数
        self.evaluator :Callable[[nn.Module, DataLoader, str], ] = None

        # 运行时参数
        # 恢复训练的检查点路径
        self.resume :str = None
        # 微调的检查点路径
        self.tuning :str = None

        # 训练的总轮数
        self.epoches :int = None
        # 上一次训练的轮数
        self.last_epoch :int = -1
        # 训练结束的轮数
        self.end_epoch :int = None

        # 是否使用混合精度训练
        self.use_amp :bool = False 
        # 是否使用EMA
        self.use_ema :bool = False 
        # 是否使用同步BN
        self.sync_bn :bool = False 
        # 梯度裁剪的最大范数
        self.clip_max_norm : float = None
        # 是否查找未使用的参数
        self.find_unused_parameters :bool = None
        # self.ema_decay: float = 0.9999
        # self.grad_clip_: Callable = None

        # 日志目录
        self.log_dir :str = './logs/'
        # 日志记录的步长
        self.log_step :int = 10
        # 输出目录
        self._output_dir :str = None
        # 打印频率
        self._print_freq :int = None 
        # 检查点保存的步长
        self.checkpoint_step :int = 1

        # 设备选择
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)

    @property
    def model(self, ) -> nn.Module:
        """获取模型实例"""
        return self._model 
    
    @model.setter
    def model(self, m):
        """设置模型实例"""
        assert isinstance(m, nn.Module), f'{type(m)} != nn.Module, please check your model class'
        self._model = m 

    @property
    def postprocessor(self, ) -> nn.Module:
        """获取后处理模块实例"""
        return self._postprocessor
    
    @postprocessor.setter
    def postprocessor(self, m):
        """设置后处理模块实例"""
        assert isinstance(m, nn.Module), f'{type(m)} != nn.Module, please check your model class'
        self._postprocessor = m 

    @property
    def criterion(self, ) -> nn.Module:
        """获取损失函数实例"""
        return self._criterion
    
    @criterion.setter
    def criterion(self, m):
        """设置损失函数实例"""
        assert isinstance(m, nn.Module), f'{type(m)} != nn.Module, please check your model class'
        self._criterion = m 

    @property
    def optimizer(self, ) -> Optimizer:
        """获取优化器实例"""
        return self._optimizer
    
    @optimizer.setter
    def optimizer(self, m):
        """设置优化器实例"""
        assert isinstance(m, Optimizer), f'{type(m)} != optim.Optimizer, please check your model class'
        self._optimizer = m 

    @property
    def lr_scheduler(self, ) -> LRScheduler:
        """获取学习率调度器实例"""
        return self._lr_scheduler
    
    @lr_scheduler.setter
    def lr_scheduler(self, m):
        """设置学习率调度器实例"""
        assert isinstance(m, LRScheduler), f'{type(m)} != LRScheduler, please check your model class'
        self._lr_scheduler = m 

    @property
    def train_dataloader(self):
        """获取训练数据加载器实例"""
        if self._train_dataloader is None and self.train_dataset is not None:
            loader = DataLoader(self.train_dataset, 
                                batch_size=self.train_batch_size, 
                                num_workers=self.num_workers, 
                                collate_fn=self.collate_fn,
                                shuffle=self.train_shuffle, )
            loader.shuffle = self.train_shuffle
            self._train_dataloader = loader

        return self._train_dataloader

    @train_dataloader.setter
    def train_dataloader(self, loader):
        """设置训练数据加载器实例"""
        self._train_dataloader = loader 

    @property
    def val_dataloader(self):
        """获取验证数据加载器实例"""
        if self._val_dataloader is None and self.val_dataset is not None:
            loader = DataLoader(self.val_dataset, 
                                batch_size=self.val_batch_size, 
                                num_workers=self.num_workers, 
                                drop_last=False,
                                collate_fn=self.collate_fn, 
                                shuffle=self.val_shuffle)
            loader.shuffle = self.val_shuffle
            self._val_dataloader = loader

        return self._val_dataloader
    
    @val_dataloader.setter
    def val_dataloader(self, loader):
        """设置验证数据加载器实例"""
        self._val_dataloader = loader 

    # TODO method
    # @property
    # def ema(self, ) -> nn.Module:
    #     if self._ema is None and self.use_ema and self.model is not None:
    #         self._ema = ModelEMA(self.model, self.ema_decay)
    #     return self._ema

    @property
    def ema(self, ) -> nn.Module:
        """获取EMA模型实例"""
        return self._ema 

    @ema.setter
    def ema(self, obj):
        """设置EMA模型实例"""
        self._ema = obj
    

    @property
    def scaler(self) -> GradScaler: 
        """获取混合精度训练的梯度缩放器实例"""
        if self._scaler is None and self.use_amp and torch.cuda.is_available():
            self._scaler = GradScaler()
        return self._scaler
    
    @scaler.setter
    def scaler(self, obj: GradScaler):
        """设置混合精度训练的梯度缩放器实例"""
        self._scaler = obj


    @property
    def val_shuffle(self):
        """获取验证数据是否打乱"""
        if self._val_shuffle is None:
            print('warning: set default val_shuffle=False')
            return False
        return self._val_shuffle

    @val_shuffle.setter
    def val_shuffle(self, shuffle):
        """设置验证数据是否打乱"""
        assert isinstance(shuffle, bool), 'shuffle must be bool'
        self._val_shuffle = shuffle

    @property
    def train_shuffle(self):
        """获取训练数据是否打乱"""
        if self._train_shuffle is None:
            print('warning: set default train_shuffle=True')
            return True
        return self._train_shuffle

    @train_shuffle.setter
    def train_shuffle(self, shuffle):
        """设置训练数据是否打乱"""
        assert isinstance(shuffle, bool), 'shuffle must be bool'
        self._train_shuffle = shuffle


    @property
    def train_batch_size(self):
        """获取训练时的批量大小"""
        if self._train_batch_size is None and isinstance(self.batch_size, int):
            print(f'warning: set train_batch_size=batch_size={self.batch_size}')
            return self.batch_size
        return self._train_batch_size

    @train_batch_size.setter
    def train_batch_size(self, batch_size):
        """设置训练时的批量大小"""
        assert isinstance(batch_size, int), 'batch_size must be int'
        self._train_batch_size = batch_size

    @property
    def val_batch_size(self):
        """获取验证时的批量大小"""
        if self._val_batch_size is None:
            print(f'warning: set val_batch_size=batch_size={self.batch_size}')
            return self.batch_size
        return self._val_batch_size

    @val_batch_size.setter
    def val_batch_size(self, batch_size):
        """设置验证时的批量大小"""
        assert isinstance(batch_size, int), 'batch_size must be int'
        self._val_batch_size = batch_size


    @property
    def output_dir(self):
        """获取输出目录"""
        if self._output_dir is None:
            return self.log_dir
        return self._output_dir

    @output_dir.setter
    def output_dir(self, root):
        """设置输出目录"""
        self._output_dir = root

    @property
    def print_freq(self):
        """获取打印频率"""
        if self._print_freq is None:
            # self._print_freq = self.log_step
            return self.log_step
        return self._print_freq

    @print_freq.setter
    def print_freq(self, n):
        """设置打印频率"""
        assert isinstance(n, int), 'print_freq must be int'
        self._print_freq = n


    # def __repr__(self) -> str:
    #     pass 



