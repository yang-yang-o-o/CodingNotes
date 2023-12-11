# [剑指 Offer 63. 股票的最大利润](https://leetcode.cn/problems/gu-piao-de-zui-da-li-run-lcof/)

---

## 题目

买卖该股票一次可能获得的最大利润是多少

---

## 思路

---

## 代码

```C++
// 单调栈，对于每个元素，记录其后的最大元素
class Solution {
public:
    int maxProfit(vector<int>& prices) {
        stack<int> S;// 单调栈栈顶存储的是当前元素后面的最大元素
        int n = prices.size();
        for(int i=n-1;i>=0;i--){
            if(S.empty() || prices[i]>=S.top())
                S.push(prices[i]);
        }
        int max_=0;
        for(int i=0;i<n;i++){
            max_ = max(max_,S.top()-prices[i]);
            if(S.top()==prices[i])// 注意这里
                S.pop();
        }
        return max_;
    }
};

// 滚动变量 + 动态规划，对于每个元素，记录其前的最小元素
class Solution {
public:
    int maxProfit(vector<int>& prices) {
        int minprice = INT_MAX; // minprice表示截止当前，出现过的最小价格
        int maxprof = 0;    // maxprofit表示在当前及其之前买卖一次能获得的最大利润 
        for(int i:prices){
            maxprof = max(maxprof,i-minprice); // 每一天都看今天卖能不能获得最大利润
            minprice = min(minprice,i); // 贪心记住最小价格
        }
        return maxprof;
    }
};
```

时间复杂度：**O( n )**  
空间复杂度：**O( 1 )**
