import torch
from typing import Dict, List, Tuple, Optional, Any, Callable  # 添加 Tuple 导入
import os
import numpy as np
from hessian_validator import HessianValidator

class HessianPointValidator(HessianValidator):
    """
    单点Hessian验证器 - PyTorch版本
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
            'energy': energy_func(test_point[0], test_point[1]).item(),
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
    
    def _compute_analytical_results(self, point: torch.Tensor, 
                                energy: Callable) -> tuple[torch.Tensor, torch.Tensor]:
        """
        同时计算解析的Force和Hessian
        
        Args:
            point: 输入点
            energy: 能量函数
            
        Returns:
            (解析Force, 解析Hessian) 的元组
        """
        x_a, x_b = point[0], point[1]
        try:
            # 检查能量函数是否有对应的解析方法
            if hasattr(energy, 'analytical_hessian') and hasattr(energy, 'analytical_force'):
                
                analytical_force = energy.analytical_force(x_a, x_b)
                analytical_hessian = energy.analytical_hessian(x_a, x_b)
                
                print("使用解析Force和Hessian函数计算")
                return analytical_force, analytical_hessian
            
            else:
                raise AttributeError("energy缺少analytical_force方法或analytical_hessian方法")
                
        except Exception as e:
            print(f"解析计算失败: {e}")
            raise
    
    def _compute_finite_difference_results(self, point: torch.Tensor, energy_func: Callable) -> tuple[torch.Tensor, torch.Tensor]:
        """
        中心差分计算Hessian - PyTorch版本
        """
        x_a, x_b = point[0], point[1]
        n = 3
        hessian = torch.zeros((n, n), device=self.device)
        
        f_force = lambda x: energy_func(x, x_b)
        f_hessian = lambda x: energy_func.analytical_force(x, x_b)
        
        def numerical_grad(f, x, h=1e-5):
            """
            用有限差分计算函数 f 在点 x 的梯度 (数值近似)
            参数:
                f : callable, 输入 torch.Tensor 返回标量 (0维张量)
                x : torch.Tensor, 需要计算梯度的点
                h : float, 差分步长
            返回:
                grad : torch.Tensor, 和 x 形状相同的梯度
            """
            grad = torch.zeros_like(x)
            # 展平索引，逐个 perturb
            for idx in range(x.numel()):
                perturb = torch.zeros_like(x)
                perturb.view(-1)[idx] = h

                f_plus  = f(x + perturb)
                f_minus = f(x - perturb)

                print(f"f_plus: {f_plus}, f_minus: {f_minus}")
                print(f"f_plus - f_minus: {f_plus - f_minus}")
                print(f"(f_plus - f_minus) / (2 * 1e-3): {(f_plus - f_minus) / (2 * h)}")

                grad.view(-1)[idx] = (f_plus - f_minus) / (2 * h)
                print(f"grad[{idx}]: {grad.view(-1)[idx]}")
            return grad

        def numerical_jacobian(f, x, h=1e-5):
            """
            用有限差分计算向量函数 f 在点 x 的 Jacobian 矩阵
            f: R^n -> R^m
            """
            x = x.clone().detach()
            y = f(x)   # y 应该是向量
            m = y.numel()
            n = x.numel()
            
            J = torch.zeros(m, n, dtype=x.dtype, device=x.device)
            
            for j in range(n):
                perturb = torch.zeros_like(x)
                perturb.view(-1)[j] = h
                f_plus = f(x + perturb).reshape(-1)
                f_minus = f(x - perturb).reshape(-1)
                J[:, j] = (f_plus - f_minus) / (2 * h)
            return J


        # 使用梯度函数的有限差分计算Hessian
        force = -numerical_grad(f_force, x_a, h=self.epsilon)

        # 使用梯度函数的有限差分计算Hessian
        hessian = -numerical_jacobian(f_hessian, x_a, h=self.epsilon)
        
        return force, hessian
    
    def _compute_automatic_diff_results(self, point: torch.Tensor, energy_func: Callable) -> tuple[torch.Tensor, torch.Tensor]:
        """
        使用PyTorch自动微分计算Hessian
        """
        # 确保point需要梯度
        x_a, x_b = point[0].clone().detach().requires_grad_(True), point[1].clone().detach().requires_grad_(True)
        n = 3
        
        # 计算梯度
        energy = energy_func(x_a, x_b)
        grad = torch.autograd.grad(energy, x_a, create_graph=True)[0]
        
        # print("自动微分计算Force:", grad)
        force = -grad

        # 计算Hessian
        hessian = torch.zeros((n, n), device=self.device)
        
        # 对于线性函数，Hessian为零，但我们仍然可以计算
        # 不需要特殊处理，因为线性函数的二阶导数就是零
        
        for i in range(n):
            # 对每个梯度分量求导
            grad_i = grad[i]
            
            # 检查梯度是否需要梯度
            if not grad_i.requires_grad:
                # 如果梯度是常数（不需要梯度），则Hessian对应行为零
                # 这种情况下，我们已经初始化hessian为零矩阵，所以不需要额外操作
                continue
                
            # 只有当梯度需要梯度时才计算二阶导数
            grad2 = torch.autograd.grad(grad_i, x_a, retain_graph=True)[0]
            hessian[i, :] = grad2
        
        return force.detach(), hessian.detach()
    
    def batch_validate(self, 
                      test_points: List[torch.Tensor],
                      energy_func: Callable = None) -> Dict[str, Any]:
        """
        批量验证多个点
        
        Args:
            test_points: 测试点列表（张量）
            energy_func: 能量函数
            
        Returns:
            批量验证结果
        """
        results = []
        passed_count = 0
        
        for i, point in enumerate(test_points):
            print(f"\n验证点 {i+1}/{len(test_points)}...")
            result = self.validate(point, energy_func)
            results.append(result)
            if result['validation_passed']:
                passed_count += 1
        
        summary = {
            'total_points': len(test_points),
            'passed_points': passed_count,
            'pass_rate': passed_count / len(test_points),
            'detailed_results': results
        }
        
        print(f"\n批量验证完成: {passed_count}/{len(test_points)} 通过 ({summary['pass_rate']:.1%})")
        return summary