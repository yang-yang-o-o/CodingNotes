# [剑指 Offer 49. 丑数](https://leetcode.cn/problems/chou-shu-lcof/description/)

---

## 题目 (中等)

给你一个整数 n ，请你找出并返回第 n 个 丑数 。  

说明：丑数是只包含质因数 2、3 和/或 5 的正整数；1 是丑数。  

示例 1：  

```markdown
输入: n = 10
输出: 12
解释: 1, 2, 3, 4, 5, 6, 8, 9, 10, 12 是前 10 个丑数。
```

提示：  

- 1 <= n <= 1690

注意：本题与[主站 264 题](https://leetcode-cn.com/problems/ugly-number-ii/)相同

---

## 思路

1、暴力解法：依次用2、3、5不断的整除num，直到不能整除，此时如果num==1了，说明num就只由2、3、5相乘得到，就是丑数

2、动态规划：每个丑数都分别乘以2、3、5，结果就是一个新的丑数

---

## 代码

```C++
class Solution {
public:
    bool isUgly(int num) {
        if(num<1)return false;// 注意这个条件
        while(num%2==0)num/=2;
        while(num%3==0)num/=3;
        while(num%5==0)num/=5;
        return num==1;
    }
};
/*
设置3个索引a, b, c，分别记录前几个数已经被乘2， 乘3， 乘5了，比如a表示前(a-1)个
数都已经乘过一次2了，下次应该乘2的是第a个数；b表示前(b-1)个数都已经乘过一次3了，
下次应该乘3的是第b个数；c表示前(c-1)个数都已经乘过一次5了，下次应该乘5的是第c个数；
*/
// 自己写的解法
class Solution {
public:
    int nthUglyNumber(int n) {
        int a=0,b=0,c=0;
        vector<int> dp(n);// dp[i]表示第i个丑数
        dp[0] = 1;
        for(int i=1;i<n;i++){// 计算n个丑数
            int va = dp[a]*2;
            int vb = dp[b]*3;
            int vc = dp[c]*5;
            dp[i] = min(va,min(vb,vc));
            if(dp[i]==va)a++;
            if(dp[i]==vb)b++;
            if(dp[i]==vc)c++;
        }
        return dp[n-1];
    }
};
```

时间复杂度：**O(logn)**  
空间复杂度：**O(1)**
