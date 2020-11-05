## Numba

- 编译型语言 
- 解释性语言 产生中间代码 每执行一次 就翻译一次
- Numba 两种模式 nopython object
- 前者不使用Python运行时并生成没有Python依赖的本机代码。本机代码是静态类型的，运行速度非常快。而对象模式使用Python对象和Python C API，而这通常不会显着提高速度。在这两种情况下，Python代码都是使用LLVM编译的。
- https://zhuanlan.zhihu.com/p/68742702 这里介绍个性能测试工具 看着不错
- numba系列了解
- 可以看这一系列 https://zhuanlan.zhihu.com/p/68852771
- 主要就是njit/parallel/向量化