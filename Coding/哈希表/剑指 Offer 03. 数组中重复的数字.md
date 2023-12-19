# [剑指 Offer 03. 数组中重复的数字](https://leetcode.cn/problems/shu-zu-zhong-zhong-fu-de-shu-zi-lcof/description/)

---

## 题目 (简单)

设备中存有 n 个文件，文件 id 记于数组 documents。若文件 id 相同，则定义为该文件存在副本。请返回任一存在副本的文件 id。

示例 1：  
```
输入：documents = [2, 5, 3, 0, 5, 0]
输出：0 或 5
```

提示：  

- 0 ≤ documents[i] ≤ n-1
- 2 <= n <= 100000

---

## 思路

原地置换：  

下标是 `[0,n-1]`，元素也是 `[0,n-1]`，如果不存在重复元素，所有元素要么下标和元素相等，要么可以通过一条链：  
`i -> nums[i] -> nums[nums[i]]`
串起来，如果出现了重复元素，那么必然存在环，不断地交换 `swap(nums[i],nums[nums[i]])`，必然会回到自身
`nums[i] == nums[nums[i]]`


---

## 代码

```C++
class Solution {
public:
    int findRepeatNumber(vector<int>& nums) {
        for(int i=0;i<nums.size();i++) // 从第一个元素开始换，换到和下标相等，然后再第二个元素
            while(nums[i]!=i){ // 只要下标和元素不等就去不停的交换
                if(nums[i] == nums[nums[i]]) // 如果当前的元素已经和要换的元素相等了，就找到重复了
                    return nums[i];
                swap(nums[i],nums[nums[i]]);
            }
        return -1;
    }
};
```

时间复杂度：**O(n)**  
空间复杂度：**O(1)**
