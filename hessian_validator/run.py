# 示例使用
def example_energy_function(x):
    """示例能量函数"""
    return np.sum(x**2) + 0.1 * np.sum(x**4)

# 点验证
point_validator = HessianPointValidator()
test_point = np.array([1.0, 2.0, 0.5])
result = point_validator.validate(test_point, example_energy_function)

# 系统验证
system_validator = HessianSystemValidator()
system_validator.set_energy_function(example_energy_function)
system_result = system_validator.validate(sampling_strategy='random', num_samples=5)