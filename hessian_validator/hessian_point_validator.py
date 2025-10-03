import numpy as np
from typing import Dict, Any, Callable, List
from hessian_validator import HessianValidator

class HessianPointValidator(HessianValidator):
    """
    单点Hessian验证器
    在特定点验证公式计算、数值差分和自动求导的一致性
    """
    
    def __init__(self, system_config: Dict[str, Any] = None):
        super().__init__(system_config)
        self.epsilon = 1e-6  # 差分步长
        self.validation_points = []
        
    def validate(self, 
                test_point: np.ndarray,
                energy_func: Callable = None,
                tolerance: float = 1e-4) -> Dict[str, Any]:
        """
        在特定点验证Hessian计算
        
        Args:
            test_point: 测试点坐标
            energy_func: 能量函数
            tolerance: 容差
            
        Returns:
            验证结果字典
        """
        if energy_func is None:
            raise ValueError("请提供能量函数")
        
        print(f"在点 {test_point} 验证Hessian...")
        
        # 三种方法计算Hessian
        analytical_hessian = self._compute_analytical_hessian(test_point, energy_func)
        finite_diff_hessian = self._compute_finite_difference_hessian(test_point, energy_func)
        auto_diff_hessian = self._compute_automatic_diff_hessian(test_point, energy_func)
        
        # 计算差异
        error_analytical_fd = self.relative_error(analytical_hessian, finite_diff_hessian)
        error_analytical_ad = self.relative_error(analytical_hessian, auto_diff_hessian)
        
        # 范数分析
        norms = {
            'analytical_norm': self.compute_norm(analytical_hessian),
            'finite_diff_norm': self.compute_norm(finite_diff_hessian),
            'auto_diff_norm': self.compute_norm(auto_diff_hessian)
        }
        
        # 验证结果
        validation_passed = (
            error_analytical_fd < tolerance and 
            error_analytical_ad < tolerance 
        )
        
        results = {
            'validation_passed': validation_passed,
            'analytical_hessian': analytical_hessian,
            'finite_diff_hessian': finite_diff_hessian,
            'auto_diff_hessian': auto_diff_hessian,
            'error_analytical_vs_finite_diff': error_analytical_fd,
            'error_analytical_vs_auto_diff': error_analytical_ad,
            'norms': norms,
            'tolerance': tolerance,
            'test_point': test_point
        }
        
        self.print_validation_summary(results)
        return results
    
    def _compute_analytical_hessian(self, point: np.ndarray, energy_func: Callable) -> np.ndarray:
        """
        公式计算Hessian
        这里需要根据你的具体能量函数实现
        """
        # 示例实现 - 需要替换为你的具体公式
        try:
            # 假设energy_func有analytical_hessian方法
            if hasattr(energy_func, 'analytical_hessian'):
                return energy_func.analytical_hessian(point)
            else:
                # 备用：使用数值差分
                print("警告：使用有限差分替代解析Hessian")
                return self._compute_finite_difference_hessian(point, energy_func)
        except Exception as e:
            print(f"解析Hessian计算失败: {e}")
            raise
    
    def _compute_finite_difference_hessian(self, point: np.ndarray, energy_func: Callable) -> np.ndarray:
        """
        中心差分计算Hessian
        """
        n = point.shape[0]
        hessian = np.zeros((n, n))
        
        # 计算梯度函数（使用有限差分）
        def gradient_func(x):
            grad = np.zeros_like(x)
            for i in range(len(x)):
                x_plus = x.copy()
                x_minus = x.copy()
                x_plus[i] += self.epsilon
                x_minus[i] -= self.epsilon
                grad[i] = (energy_func(x_plus) - energy_func(x_minus)) / (2 * self.epsilon)
            return grad
        
        # 使用梯度函数的有限差分计算Hessian
        base_grad = gradient_func(point)
        
        for i in range(n):
            point_perturbed = point.copy()
            point_perturbed[i] += self.epsilon
            grad_perturbed = gradient_func(point_perturbed)
            hessian[i, :] = (grad_perturbed - base_grad) / self.epsilon
        
        return hessian
    
    def _compute_automatic_diff_hessian(self, point: np.ndarray, energy_func: Callable) -> np.ndarray:
        """
        自动求导计算Hessian
        使用JAX或其他自动微分库
        """
        try:
            # 尝试导入JAX
            import jax
            import jax.numpy as jnp
            
            # 将函数转换为JAX可微分函数
            energy_jax = jax.jit(energy_func)
            
            # 计算Hessian
            hessian_func = jax.jacfwd(jax.jacrev(energy_jax))
            hessian = np.array(hessian_func(point))
            return hessian
            
        except ImportError:
            print("JAX未安装，使用有限差分替代自动微分")
            return self._compute_finite_difference_hessian(point, energy_func)
    
    def batch_validate(self, 
                      test_points: List[np.ndarray],
                      energy_func: Callable = None) -> Dict[str, Any]:
        """
        批量验证多个点
        
        Args:
            test_points: 测试点列表
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