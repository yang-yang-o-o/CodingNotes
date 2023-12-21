# [剑指 Offer 13. 机器人的运动范围](https://leetcode.cn/problems/ji-qi-ren-de-yun-dong-fan-wei-lcof/description/)

---

## 题目

家居整理师将待整理衣橱划分为 m x n 的二维矩阵 grid，其中 grid[i][j] 代表一个需要整理的格子。整理师自 grid[0][0] 开始 逐行逐列 地整理每个格子。  

整理规则为：在整理过程中，可以选择 向右移动一格 或 向下移动一格，但不能移动到衣柜之外。同时，不需要整理 digit(i) + digit(j) > cnt 的格子，其中 digit(x) 表示数字 x 的各数位之和。  

请返回整理师 总共需要整理多少个格子。  

示例 1：  
```
输入：m = 4, n = 7, cnt = 5
输出：18
```

提示：  

- 1 <= n, m <= 100
- 0 <= cnt <= 20

---

## 思路

---

## 代码

```C++
class Solution {
public:
    int get(int x){
        int res=0;
        while(x>0){// 注意这种写法
            res += x%10;
            x /= 10;
        }
        return res;
    }
    int movingCount(int m, int n, int k) {
        if(!k)return 1;
        vector<vector<int>> mp(m,vector<int>(n,0));// 标记位置是否访问过
        vector<vector<int>> direction{{1,0},{0,1}};
        queue<pair<int,int>> Q;
        Q.push({0,0});
        mp[0][0] = 1;
        int res=1;
        while(!Q.empty()){
            pair<int,int> p = Q.front();Q.pop();
            int x = p.first;
            int y = p.second;
            for(int i=0;i<2;i++){// 两个方向，相当于两个子树
                int newx = x + direction[i][0];
                int newy = y + direction[i][1];
                if(newx<m && newy<n && mp[newx][newy]==0 && get(newx)+get(newy)<=k){// 注意这个判断条件
                    mp[newx][newy] = 1;// 注意设置已访问的语句为什么放这里
                    Q.push({newx,newy});
                    res++;
                }
            }
        }
        return res;
    }
};
```
