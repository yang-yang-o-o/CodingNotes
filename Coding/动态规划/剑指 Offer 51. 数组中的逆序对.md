# [剑指 Offer 51. 数组中的逆序对](https://leetcode.cn/problems/shu-zu-zhong-de-ni-xu-dui-lcof/)

---

## 题目 (困难)

在股票交易中，如果前一天的股价高于后一天的股价，则可以认为存在一个「交易逆序对」。请设计一个程序，输入一段时间内的股票交易记录 record，返回其中存在的「交易逆序对」总数。  

示例 1:  
```
输入：record = [9, 7, 5, 4, 6]  
输出：8  
解释：交易中的逆序对为 (9, 7), (9, 5), (9, 4), (9, 6), (7, 5), (7, 4), (7, 6), (5, 4)。  
```

限制：  

- 0 <= record.length <= 50000

---

## 思路

归并排序

---

## 代码

```C++
class Solution {
public:
    int mergesort(vector<int>& nums,vector<int>& tmp,int L,int R){
        if(L>=R) // 只有一个元素，逆序对为0
            return 0;
        int mid = (L+R)/2;
        int res = mergesort(nums,tmp,L,mid) + mergesort(nums,tmp,mid+1,R);// // 左边的逆序对+右边的逆序对（这里要是mid和mid+1）
        int i = L;
        int j = mid+1;
        int p = L;
        while(i<=mid && j<=R){
            if(nums[i]<=nums[j]){// 这里一定要是小于等于
                tmp[p++] = nums[i++];
                res += (j-(mid+1));// 左指针比右指针小，那么右指针左边的所有元素都比左指针小（这里很巧妙，因为每次是较小的元素放到tmp里，如果nums[j-1]比nums[i]大就矛盾了），每个元素就都能和左指针构成一个逆序对（注意这里不算j，所以减mid+1即可）
            }
            else
                tmp[p++] = nums[j++];
        }
        while(i<=mid){
            tmp[p++] = nums[i++];
            res += (j-(mid+1)); // 左指针没有处理完，那么左指针剩下的每个元素都能和右边的所有元素分别组一个逆序对（左边剩下的每个元素都大于右边的所有元素），注意这里j已经等于R+1了，不能算j，所以减mid+1即可
        }
        while(j<=R){
            tmp[p++] = nums[j++];
        }
        copy(tmp.begin()+L,tmp.begin()+R+1,nums.begin()+L);
        return res;
    }
    int reversePairs(vector<int>& nums) {
        int n = nums.size();
        vector<int> tmp(n);
        return mergesort(nums,tmp,0,n-1);
    }
};
```

时间复杂度：**O( nlogn )**  
空间复杂度：**O( n )**
