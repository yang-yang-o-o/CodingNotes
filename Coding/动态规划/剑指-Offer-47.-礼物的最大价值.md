# [剑指 Offer 47. 礼物的最大价值](https://leetcode.cn/problems/li-wu-de-zui-da-jie-zhi-lcof/)

---

## 题目 (中等)

现有一个记作二维矩阵 `frame` 的珠宝架，其中 `frame[i][j]` 为该位置珠宝的价值。拿取珠宝的规则为：  

- 只能从架子的左上角开始拿珠宝
- 每次可以移动到右侧或下侧的相邻位置
- 到达珠宝架子的右下角时，停止拿取
注意：珠宝的价值都是大于 `0` 的。除非这个架子上没有任何珠宝，比如 `frame = [[0]]`。  

示例 1:  

```markdown
输入: frame = [[1,3,1],[1,5,1],[4,2,1]]  
输出: 12  
解释: 路径 1→3→5→2→1 可以拿到最高价值的珠宝  
```

提示：  

- 0 < frame.length <= 200
- 0 < frame[0].length <= 200

---

## 思路

动态规划

---

## 代码

```C++
class Solution {
public:
    int maxValue(vector<vector<int>>& grid) {
        int m = grid.size();
        int n = grid[0].size();
        vector<vector<int>> dp(m,vector<int>(n,0));// dp[i][j]表示从源点到达[i][j]位置能得到的最大礼物价值
        for(int i=0;i<m;i++)
            for(int j=0;j<n;j++){
                if(i==0 && j==0)// 源点
                    dp[i][j] = grid[i][j];
                else if(i==0)   // 第0行
                    dp[i][j] = dp[i][j-1] + grid[i][j];
                else if(j==0)   // 第0列
                    dp[i][j] = dp[i-1][j] + grid[i][j];
                else
                    dp[i][j] = max(dp[i-1][j],dp[i][j-1]) + grid[i][j];
            }
        return dp[m-1][n-1];
    }
};
```

时间复杂度：**O( mn )**  
空间复杂度：**O( mn )**
