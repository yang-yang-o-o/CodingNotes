# [剑指 Offer 53 - I. 在排序数组中查找数字](https://leetcode.cn/problems/find-first-and-last-position-of-element-in-sorted-array/description/)

---

## 题目

在一个非递减的数组`nums`中，找出目标值`target`出现的`开始位置`和`结束位置`。  

示例 1：  
输入：nums = [5,7,7,8,8,10], target = 8  
输出：[3,4]  

示例 2：  
输入：nums = [5,7,7,8,8,10], target = 6  
输出：[-1,-1]  

示例 3：  
输入：nums = [], target = 0  
输出：[-1,-1]  

---

## 思路

二分查找：执行两次二分查找，分别查找`第一个大于等于target`的下标、`第一个大于target`的下标。

---

## 代码

```C++
class Solution {
public:
    int binarysearchleft(vector<int>& nums,int target){
        int l = 0, r = nums.size()-1, ans = nums.size();// 注意ans的初始值
        while(l<=r){
            int mid = (l+r)/2;
            if(nums[mid] >= target){ // 唯一区别
                r = mid -1;
                ans = mid; // 只要大于等于target，就记录这个值，然后去左边查找，最终找到的就是第一个大于等于target的值
            }
            else
                l = mid+1;
        }
        return ans;
    }
    int binarysearchright(vector<int>& nums,int target){
        int l = 0, r = nums.size()-1, ans = nums.size();
        while(l<=r){
            int mid = (l+r)/2;
            if(nums[mid] > target){ // 唯一区别
                r = mid -1;
                ans = mid; // 只要大于target，就记录这个值，然后去左边查找，最终找到的就是第一个大于target的值
            }
            else
                l = mid+1;
        }
        return ans;
    }
    vector<int> searchRange(vector<int>& nums, int target) {
        int l = binarysearchleft(nums,target);// 查找第一个大于等于target的下标
        int r = binarysearchright(nums,target)-1;// 查找第一个大于target的下标，再减1，得到最后一个大于等于target的下标
        if(l<=r)
            return vector<int>{l,r};
        return vector<int>{-1,-1};
    }
};

// 标准二分查找
template<class T>
int binary_search_nonrecursion(vector<T> List,T n)
{
    int start = 0, end = List.size()-1;
    while(start <= end)
    {
        int mid = (start + end)/2;
        if(List[mid] < n)
            start = mid + 1;
        else if(List[mid] > n)
            end = mid - 1;
        else 
            return mid;
    }
    return -1;
}
```

时间复杂度：**O(logn)**  
空间复杂度：**O(1)**
