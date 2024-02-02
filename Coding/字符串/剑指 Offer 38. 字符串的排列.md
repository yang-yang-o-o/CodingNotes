# [剑指 Offer 38. 字符串的排列](https://leetcode.cn/problems/zi-fu-chuan-de-pai-lie-lcof/description/)

---

## 题目 (中等)

某店铺将用于组成套餐的商品记作字符串 goods，其中 goods[i] 表示对应商品。请返回该套餐内所含商品的 全部排列方式 。  

返回结果 无顺序要求，但不能含有重复的元素。  

示例 1:  

```markdown
输入：goods = "agew"
输出：["aegw","aewg","agew","agwe","aweg","awge","eagw","eawg","egaw","egwa","ewag","ewga","gaew","gawe","geaw","gewa","gwae","gwea","waeg","wage","weag","wega","wgae","wgea"]
```

提示：  

- 1 <= goods.length <= 8

---

## 思路

回溯，使用哈希集合去重

---

## 代码

```C++
class Solution {
public:
    void fullarray(unordered_set<string>& sets,string& tmp,int idx){
        if(idx == tmp.size()){
            if(sets.find(tmp)==sets.end())
                sets.insert(tmp);
            return;
        }
        for(int i=idx;i<tmp.size();i++){
            swap(tmp[idx],tmp[i]);
            fullarray(sets,tmp,idx+1);
            swap(tmp[idx],tmp[i]);
        }
    }
    vector<string> permutation(string s) {
        unordered_set<string> sets;
        fullarray(sets,s,0);
        vector<string> res;
        for(string i:sets)
            res.push_back(i);
        return res;
    }
};
```
