# [剑指 Offer 45. 把数组排成最小的数](https://leetcode.cn/problems/ba-shu-zu-pai-cheng-zui-xiao-de-shu-lcof/description/)

---

## 题目 (中等)

闯关游戏需要破解一组密码，闯关组给出的有关密码的线索是：  

- 一个拥有密码所有元素的非负整数数组 password
- 密码是 password 中所有元素拼接后得到的最小的一个数

请编写一个程序返回这个密码。

示例 1:  
```
输入: password = [15, 8, 7]
输出: "1578"
```

示例 2:  
```
输入: password = [0, 3, 30, 34, 5, 9]
输出: "03033459"
```

提示:  

- 0 < password.length <= 100

说明:  

- 输出结果可能非常大，所以你需要返回一个字符串而不是整数
- 拼接起来的数字可能会有前导 0，最后结果不需要去掉前导 0

---

## 思路

---

## 代码

```C++
class Solution {
public:
    string minNumber(vector<int>& nums) {
        vector<string> strs;
        for(int i = 0; i < nums.size(); i++)
            strs.push_back(to_string(nums[i]));
        quickSort(strs, 0, strs.size() - 1);
        string res;
        for(string s : strs)
            res.append(s);
        return res;
    }
private:
    void quickSort(vector<string>& strs, int l, int r) {
        if(l >= r) return;
        int i = l, j = r;
        while(i < j) {
            while(strs[j] + strs[l] >= strs[l] + strs[j] && i < j) j--;
            while(strs[i] + strs[l] <= strs[l] + strs[i] && i < j) i++;
            swap(strs[i], strs[j]);
        }
        swap(strs[i], strs[l]);
        quickSort(strs, l, i - 1);
        quickSort(strs, i + 1, r);
    }
};
// 自己写的解法             快速排序
class Solution {
public:
    bool less_(int a,int b){
        return to_string(a)+to_string(b) < to_string(b)+to_string(a);// 如果a放在b的前面更小，就把a放在b的前面，注意这个的传递性是需要证明的
    }
    void quicksort(vector<int>& nums,int L,int R){
        if(L>=R)
            return;
        int p = rand()%(R-L+1)+L;
        swap(nums[p],nums[R]);
        int i=L-1;
        for(int j=L;j<R;j++)
            if(less_(nums[j],nums[R]))
                swap(nums[j],nums[++i]);
        swap(nums[R],nums[++i]);
        quicksort(nums,L,i-1);
        quicksort(nums,i+1,R);
    }
    string minNumber(vector<int>& nums) {
        srand((unsigned)time(nullptr));
        quicksort(nums,0,nums.size()-1);
        string res ="";
        for(int i:nums)
            res += to_string(i);
        return res;
    }
};
```
