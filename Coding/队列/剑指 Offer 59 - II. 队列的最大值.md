# [剑指 Offer 59 - II. 队列的最大值](https://leetcode.cn/problems/dui-lie-de-zui-da-zhi-lcof/)

---

## 题目

请设计一个自助结账系统，该系统需要通过一个队列来模拟顾客通过购物车的结算过程，需要实现的功能有：  

get_max()：获取结算商品中的最高价格，如果队列为空，则返回 -1  
add(value)：将价格为 value 的商品加入待结算商品队列的尾部  
remove()：移除第一个待结算的商品价格，如果队列为空，则返回 -1  
注意，为保证该系统运转高效性，以上函数的均摊时间复杂度均为 O(1)  

示例 1：
```
输入: 
["Checkout","add","add","get_max","remove","get_max"]
[[],[4],[7],[],[],[]]

输出: [null,null,null,7,4,7]
```
示例 2：
```
输入: 
["Checkout","remove","get_max"]
[[],[],[]]

输出: [null,-1,-1]
```

提示：  

1 <= get_max, add, remove 的总操作数 <= 10000
1 <= value <= 10^5

---

## 思路

用一个双端队列来实现一个单调队列，单调队列首元素为当前结算队列中的最大值

---

## 代码

```C++
class MaxQueue {
public:
    deque<int> tmp; // 双端队列用于实现单调队列
    queue<int> q; // 结算队列
    MaxQueue() {

    }
    
    int max_value() {
        if(q.empty())
            return -1;
        return tmp.front();
    }
    
    void push_back(int value) {
        q.push(value);
        while(!tmp.empty() && tmp.back()<value) // 重点
            tmp.pop_back();
        tmp.push_back(value);
    }
    
    int pop_front() {
        if(q.empty())
            return -1;
        int res = q.front();
        q.pop();
        if(res == tmp.front())
            tmp.pop_front();
        return res;
    }
};
```

时间复杂度：**O(n)**  
空间复杂度：**O(1)**
