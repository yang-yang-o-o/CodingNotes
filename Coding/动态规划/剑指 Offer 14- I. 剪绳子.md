# [剑指 Offer 14- I. 剪绳子](https://leetcode.cn/problems/jian-sheng-zi-lcof/)

---

## 题目 (中等)

现需要将一根长为正整数 `len` 的绳子剪为若干段，每段长度均为正整数。请返回每段绳子长度的`最大乘积`是多少。  

示例 1：  
输入: len = 12  
输出: 81  

提示：  
2 <= len <= 58  
注意：本题与[主站 343 题相同](https://leetcode-cn.com/problems/integer-break/)

---

## 思路

动态规划

---

## 代码

```C++
class Solution {
public:
    int cuttingRope(int n) {
        vector<int> dp(n+1,0); // dp[i]表示长度为i的绳子剪完后的最大乘积
        for (int i = 2; i <= n; i++) { // 枚举每种长度
            for (int j = 1; j < i; j++) // 从之前的长度转移得到当前
                dp[i] = max(dp[i], max(j*(i-j), j*dp[i-j])); // j*(i-j)表示剪了j后剩下的不剪，j*dp[i-j]表示剪了j后剩下的剪
        }
        return dp[n];
    }
};
```

时间复杂度：**O( n² )**  
空间复杂度：**O( n )**
