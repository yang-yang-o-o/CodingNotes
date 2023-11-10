# [剑指 Offer 40. 最小的k个数](https://leetcode.cn/problems/smallest-k-lcci/description/)

---

## 题目

找出数组中最小的`k`个数。以任意顺序返回这`k`个数均可。  

示例：  
输入： arr = [1,3,5,7,2,4,6,8], k = 4  
输出： [1,2,3,4]  

---

## 思路

快速排序

---

## 代码

```C++
// 基于快速排序的选择算法
class Solution {
public:
    void quicksort(vector<int>& nums,int L,int R,int k){
        if(L>=R) // 递归终止条件
            return;
        int P = rand()%(R-L+1)+L; // 随机选择主元
        swap(nums[P],nums[R]); // 主元交换到最右边
        int pivort = nums[R];
        int i = L-1;
        for(int j=L;j<=R-1;j++) // 从左开始遍历
            if(nums[j]<=pivort)
                swap(nums[++i],nums[j]); // 如果当前元素小于等于主元，就换到左边
        swap(nums[++i],nums[R]); // 主元放到下标i处，下标i左边的元素都比主元小，下标i右边的元素都比主元大
        int num = i-L+1; // 计算小于主元的个数，也就是最小的num个数
        if(num==k) // 找到了最小的k个数
            return;
        else if(num<k)
            quicksort(nums,i+1,R,k-num); // 还不够，去右边找最小的k-num个数
        else
            quicksort(nums,L,i-1,k); // 多了，去左边找最小的k个数
    }
    vector<int> getLeastNumbers(vector<int>& arr, int k) {
        srand((unsigned)time(nullptr)); // 设置随机数种子
        quicksort(arr,0,arr.size()-1,k);
        vector<int> res;
        for(int i=0;i<k;i++) // 获取最小的前k个数
            res.push_back(arr[i]);
        return res;
    }
};

// 原始快速排序
class Solution {
public:
    int findKthLargest(vector<int>& nums, int k) {
        srand((unsigned)time(nullptr));
        Qsort(nums,0,nums.size()-1);
        return nums[nums.size()-k];
    }
    void Qsort(vector<int>& nums,int L, int R){
        if(L>=R)
            return;
        int P = rand() % (R-L+1) + L;
        swap(nums[P],nums[R]);
        int pivort = nums[R];
        int i = L-1;
        for(int j=L;j<=R-1;j++)
            if(nums[j]<=pivort){
                i++;
                swap(nums[i],nums[j]);
            }
        swap(nums[++i],nums[R]);
        Qsort(nums,L,i-1);
        Qsort(nums,i+1,R);
    }
};
```

时间复杂度：**O(nlogn)**  
空间复杂度：**O(logn)**
