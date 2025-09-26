import newton

# 列出包里所有属性（模块、子包、类等）
print(dir(newton))

# 列出所有可导入的子模块/子包
import pkgutil
for module in pkgutil.iter_modules(newton.__path__):
    print(module.name)  