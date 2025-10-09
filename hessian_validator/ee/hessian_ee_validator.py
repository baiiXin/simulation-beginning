import torch
from typing import Dict, List, Tuple, Optional, Any, Callable  # 添加 Tuple 导入
import os
import numpy as np
from hessian_validator import HessianValidator

class HessianEEValidator(HessianValidator):
    """
    两点Hessian验证器 - PyTorch版本
    利用PyTorch自动微分进行精确验证
    """
    
    def __init__(self, system_config: Dict[str, Any] = None, tolerance: float = 1e-4, device: str = None):
        super().__init__(system_config, device)
        self.epsilon = tolerance  # 差分步长
        self.validation_points = []
        
    def validate(self, 
                test_point: torch.Tensor,
                energy_func: Callable = None,
                tolerance: float = 1e-4) -> Dict[str, Any]:
        """
        在特定点验证Hessian计算
        
        Args:
            test_point: 测试点坐标张量
            energy_func: 能量函数
            tolerance: 容差
            
        Returns:
            验证结果字典
        """
        # 确保测试点在正确的设备上
        test_point = self.to_device(test_point).requires_grad_(True)
        
        if energy_func is None:
            energy_func = self.energy_function
        if energy_func is None:
            raise ValueError("请提供能量函数")
        
        print(f"在点 {test_point.cpu().detach().numpy()} 验证Hessian...")
        
        # 三种方法计算Hessian
        analytical_results = self._compute_analytical_results(test_point, energy_func)
        finite_diff_results = self._compute_finite_difference_results(test_point, energy_func)
        auto_diff_results = self._compute_automatic_diff_results(test_point, energy_func)
        
        # 计算差异
        error_analytical_fd = self.relative_error(analytical_results[1], finite_diff_results[1])
        error_analytical_ad = self.relative_error(analytical_results[1], auto_diff_results[1])
        error_fd_ad = self.relative_error(finite_diff_results[1], auto_diff_results[1])
        
        # 验证结果
        validation_passed = (
            error_analytical_fd < tolerance and 
            error_analytical_ad < tolerance and 
            error_fd_ad < tolerance
        )
        
        results = {
            'validation_passed': validation_passed,
            'energy': energy_func(test_point).item(),
            'analytical_hessian': analytical_results[1],
            'finite_diff_hessian': finite_diff_results[1],
            'auto_diff_hessian': auto_diff_results[1],
            'analytical_force': analytical_results[0],
            'finite_diff_force': finite_diff_results[0],
            'auto_diff_force': auto_diff_results[0],
            'error_analytical_vs_finite_diff': error_analytical_fd,
            'error_analytical_vs_auto_diff': error_analytical_ad,
            'error_finite_diff_vs_auto_diff': error_fd_ad,
            'tolerance': tolerance,
            'test_point': test_point.cpu().detach().numpy()
        }
        
        self.print_validation_summary(results)
        return results
    
    def _compute_analytical_results(self, points: torch.Tensor, 
                                energy: Callable) -> tuple[torch.Tensor, torch.Tensor]:
        """
        同时计算解析的Force和Hessian
        
        Args:
            point: 输入点
            energy: 能量函数
            
        Returns:
            (解析Force, 解析Hessian) 的元组
        """
        try:
            # 检查能量函数是否有对应的解析方法
            if hasattr(energy, 'analytical_hessian') and hasattr(energy, 'analytical_force'):
                
                analytical_force = energy.analytical_force(points)
                analytical_hessian = energy.analytical_hessian(points)
                
                print("使用解析Force和Hessian函数计算")
                return analytical_force, analytical_hessian
            
            else:
                raise AttributeError("energy缺少analytical_force方法或analytical_hessian方法")
                
        except Exception as e:
            print(f"解析计算失败: {e}")
            raise
    
    def _compute_finite_difference_results(self, points: torch.Tensor, energy_func: Callable) -> tuple[torch.Tensor, torch.Tensor]:
        """
        中心差分计算Hessian - PyTorch版本
        """
        h = 1e-3
        N = points.shape[0]
        n = points[0].shape[0]

        # force
        # 展平索引，逐个 perturb
        grad = torch.zeros_like(points)
        for idx in range(N):
            for space_x in range(n):
                perturb = torch.zeros_like(points)
                perturb[idx, space_x] = h
                f_plus  = energy_func(points + perturb)
                f_minus = energy_func(points - perturb)
                grad[idx, space_x] = (f_plus - f_minus) / (2 * h)

        # hessian
        Jacobi = torch.zeros((N, N, n, n), device=self.device)
        for jdx in range(N):
            for space_jx in range(n):
                perturb = torch.zeros_like(points)
                perturb[jdx, space_jx] = h
                f_plus  = energy_func.analytical_force(points + perturb)
                f_minus = energy_func.analytical_force(points - perturb)
                Jacobi[:, jdx, :, space_jx] = (f_plus - f_minus) / (2 * h)
        
        return -grad, -Jacobi
    
    def _compute_automatic_diff_results(self, points: torch.Tensor, energy_func: Callable) -> tuple[torch.Tensor, torch.Tensor]:
        """
        使用PyTorch自动微分计算Hessian
        """
        N = points.shape[0]
        n = points[0].shape[0]

        points.requires_grad_(True)
        # 计算梯度
        energy = energy_func(points)
        grad = torch.autograd.grad(energy, points, create_graph=True)[0]

        # 初始化 Hessian
        hessian_auto = torch.zeros((N, N, n, n), device=self.device)

        # 逐分量求二阶导数
        for i in range(N):
            for a in range(n):
                # grad[i, a] 仍在计算图中，因此可以再次求导
                grad2 = torch.autograd.grad(
                    grad[i, a], points, retain_graph=True, create_graph=False
                )[0]
                hessian_auto[i, :, :, a] = grad2
        
        return -grad.detach(), hessian_auto.detach()
    
