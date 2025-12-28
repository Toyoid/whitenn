# Examples to show WhiteNN

## Example 1: `hello_whitenn.wnn`

目标：让读者一眼看到 WhiteNN 的核心闭环：构图 → 求导 → 解释 → 更新。

内容：一个标量二次函数最小化：
𝐿 = (𝑤−3)^2

内建 rule square 或用 * 表达。

explain level 1 & 2 输出计算图与链式法则展开。

向使用者展示：1 次 step 前后 w 的变化，以及 explain 输出。

## Example 2: `mystery_no_rule.wnn`

目标：展示 Rule 强制机制（“没有导数就不能 derive” 的报错样例），证明 WhiteNN 的核心创新：函数必须携带局部数学知识，不是靠库黑盒。

内容：写一个 rule mystery(x) 但只提供 forward、不提供 d/dx；在 graph 里用它，然后 derive。

期望输出：编译/解释阶段报错（示例）：

“Missing derivative clause(s): x in rule 'mystery'”

## Example 3: mlp_one_step.wnn

展示结合 MLP 使用 rule 自定义激活函数的例子，包括ReLU，Tanh 和 Sigmoid。并同时涵盖双层MLP的单步训练过程。

## Example 4: linear_regression.wnn

展示使用 MLP 进行线性回归的完整训练过程，包括数据生成、模型定义、前向传播、损失计算、梯度求导、参数更新和训练循环。

注：如需在 host 端累计/平均 loss，需要先从 graph 中取回值，例如：

```
fetch L_val = L;
print(L_val);
```

## Example 5: transformer_block.wnn

展示一个简化的 Transformer block：单头自注意力 + 前馈网络 + 残差连接，使用 softmax/transpose 等向量算子（无 batching）。

## Example 6: transformer_sort_teacher.wnn

展示使用 teacher MLP 生成排序目标，训练注意力矩阵输出排序的多分类任务（T=6, D=4），并包含不同 token 数量的测试样例。
