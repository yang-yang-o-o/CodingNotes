# 目录

- ## [1、递归](#customname1)

- ## [2、动态规划](#customname2)

- ## [3、队列](#customname3)

- ## [4、二分](#customname4)

- ## [5、哈希表](#customname5)

- ## [6、回溯](#customname6)

- ## [7、链表](#customname7)

- ## [8、排序](#customname8)

- ## [9、树](#customname9)

- ## [10、数学](#customname10)

- ## [11、数组](#customname11)

- ## [12、双指针](#customname12)

- ## [13、贪心](#customname13)

- ## [14、图](#customname14)

- ## [15、位运算](#customname15)

- ## [16、栈](#customname16)

- ## [17、字符串](#customname17)

---

## 1、递归 {#customname1}

### [面试题_08.06._汉诺塔问题](../Coding\递归\面试题_08.06._汉诺塔问题.md)

```C++
/*
    递归，核心：move(A.size(),A,B,C); // 将 A 上的n个，借助B，倒序移动到C
*/
class Solution {
public:
    void move(int n,vector<int>& A,vector<int>& B,vector<int>& C){
        if(n==1){ // 如果只剩1个，直接从A移动到C
            C.push_back(A.back());
            A.pop_back();
            return;
        }
        move(n-1,A,C,B); // 将A的n-1个借助C移动到B
        C.push_back(A.back()); // 将A的剩下一个直接移动到C
        A.pop_back();
        move(n-1,B,A,C); // 将B的n-1个借助A移动到C
    }
    void hanota(vector<int>& A, vector<int>& B, vector<int>& C) {
        move(A.size(),A,B,C); // 将 A 上的n个，借助B，倒序移动到C
    }
};
// 时间复杂度：**O( 2ⁿ−1 )**  
// 空间复杂度：**O( 1 )**
```

### [面试题_16.11.跳水板](../Coding\递归\面试题_16.11.跳水板.md)

```C++
/*
    总的有 `k+1` 种情况，`用0块长的`一直到`用k块长的`，`用k块长的`等于`用k-1块长的`加上一个`长的和短的之差`
*/
class Solution {
public:
    vector<int> divingBoard(int shorter, int longer, int k) {
        vector<int> res;
        if (k == 0) return res;
        int min_ = k * shorter; // 选0块长的
        if (shorter == longer) return vector<int>{min_};
        res.push_back(min_);
        for (int i = 1; i <= k; i++) { // 选k块长的可以从选k-1块长的转移过来
            min_ += longer - shorter; // 用k块长的 等于 用k-1块长的 加上一个 长的和短的之差
            res.push_back(min_);
        }
        return res;
    }
};
// 时间复杂度：**O(n)**  
// 空间复杂度：**O(n)**
```

### [剑指_Offer_16.数值的整数次方](../Coding\递归\剑指_Offer_16.数值的整数次方.md)

```C++
/*
    快速幂 + 迭代：  
    每次x = x²，然后n = n/2；然后在为1的那一位时把x乘到结果上
    
    x的b次方x^b，b可以用二进制表示，不断地将x = x²，b >>= 1，不断地考虑b的最后一位，此时x也就是对应的一个乘积项  
    例如当b为9时，b可以写为 2^3 + 2^0 ，也就对应二进制的 1001，第一位为1，res乘上x就相当于乘上x^(2^0)，然后b右移三次最后一位又为1，此时res乘上的是x^(2^3)，最终res = x^(2^0) * x^(2^3) = x^(2^0 + 2^3) = x^9  
*/
class Solution {
public:
    double myPow(double x, int n) {
        if(x==0)return 0;

        long b=n;
        double res=1;
        if(b<0){ // 处理负数次方
            x = 1/x;
            b = -b;
        }

        while(b>0){
            if((b&1)==1) res *= x;// 在b为1的那一位时，将当前的x乘到结果上，比如b为9，也就是1001，
                                // 在第一位时和第4位时将x乘到res上，也就是res = x * x^8 ,中间的两次while没有执行这个if是因为要攒起来得到x^8
            x = x*x;     // 每次x = x²，然后n = n/2；就能快速求幂
            b >>= 1;
        }
        return res;
    }
};
// 时间复杂度：**O( logn )** 即为对 n 进行二进制拆分的时间复杂度。  
// 空间复杂度：**O( 1 )**
```

## 2、动态规划 {#customname2}

### [62.不同路径](../Coding\动态规划\62.不同路径.md)

```C++
/*
    每次只能向下或者向右移动一步，
    动态规划，`f[i][j]` 表示到位置 `(i,j)` 的总路径数  
    状态转移：`f[i][j] = f[i-1][j] + f[i][j-1]`
*/
class Solution {
public:
    int uniquePaths(int m, int n) {
        vector<vector<int>> f(m,vector<int>(n));// mxn
        // f[i][0] = f[0][j] = 1
        for(int i=0;i<m;++i)
            f[i][0] = 1;
        for(int j=0;j<n;++j)
            f[0][j] = 1;
        // 动态规划
        for(int i=1;i<m;++i)
            for(int j=1;j<n;++j)
                f[i][j] = f[i-1][j] + f[i][j-1];// 状态转移
        return f[m-1][n-1];
    }
};
// 时间复杂度：**O( mn )**  
// 空间复杂度：**O( mn )**
```

### [63.不同路径-II](../Coding\动态规划\63.不同路径-II.md)

```C++
/*
    每次只能向下或者向右移动一步，存在障碍物
    如果当前是障碍物，则当前置0，右侧不用更新；当前不是障碍物，当前更新，右侧需要更新
*/
class Solution {
public:
    int uniquePathsWithObstacles(vector<vector<int>>& obstacleGrid) {
        int n = obstacleGrid.size(), m = obstacleGrid.at(0).size();
        vector<int> f(m); // 状态，m为列数，这里使用了滚动数组

        f[0] = (obstacleGrid[0][0] == 0); // 左上角，1表示存在障碍物，0表示不存在
        for (int i = 0; i < n; ++i) { // 行
            for (int j = 0; j < m; ++j) { // 列
                if (obstacleGrid[i][j] == 1) { // 如果存在障碍物
                    f[j] = 0;
                    continue;
                } else if (j - 1 >= 0) { // 如果左边不存在障碍物，左边必然可以到达
                    f[j] += f[j - 1]; // 注意 += 考虑了左边和上边
                }
            }
        }

        return f.back();
    }
};
// 时间复杂度：**O( mn )**  
// 空间复杂度：**O( m )**
```

### [64.最小路径和](../Coding\动态规划\64.最小路径和.md)

```C++
/*
    每次只能向下或者向右移动一步
    dp[i][j] 表示从左上角出发到 (i,j) 位置的最小路径和
    选择左或上路径和最小的，来到达当前位置
    状态更新：dp[i][j] = min(dp[i-1][j],dp[i][j-1]) + grid[i][j];
*/
class Solution {
public:
    int minPathSum(vector<vector<int>>& grid) {
        if (grid.size() == 0 || grid[0].size() == 0)
            return 0;
        int rows = grid.size() , columns = grid[0].size();
        auto dp = vector<vector<int>> (rows,vector<int>(columns)); // dp[i][j] 表示从左上角出发到 (i,j) 位置的最小路径和
        // 第0行和第0列
        dp[0][0] = grid[0][0];
        for (int i = 1; i < rows; ++i)
            dp[i][0] = dp[i-1][0] + grid[i][0];
        for (int j = 1; j < columns; ++j)
            dp[0][j] = dp[0][j-1] + grid[0][j];
        // 其它行列
        for (int i = 1; i < rows; ++i)
            for(int j = 1; j < columns; ++j)
                dp[i][j] = min(dp[i-1][j],dp[i][j-1]) + grid[i][j];

        return dp[rows-1][columns-1];
    }
};
// 时间复杂度：**O( mn )**  
// 空间复杂度：**O( mn )**
```

### [221.最大正方形](../Coding\动态规划\221.最大正方形.md)

```C++
/*
    求只包含 1 的最大正方形的面积
    dp(i,j) 表示以 (i,j) 为右下角，且只包含 1 的正方形的边长最大值。
    状态转移（三者最小的加1）：dp[i][j] = min( dp[i-1][j-1] , min( dp[i-1][j] , dp[i][j-1] ) ) + 1
*/
class Solution {
public:
    int maximalSquare(vector<vector<char>>& matrix) {
        int m = matrix.size();
        int n = matrix[0].size();
        vector<vector<int>> dp(m,vector<int>(n,0)); // 状态矩阵，dp(i,j) 表示以 (i,j) 为右下角，
                                                            // 且只包含 1 的正方形的边长最大值。
        int maxl = 0; // 最大边长

        for(int i=0;i<m;i++)
            for(int j=0;j<n;j++) // 遍历每个元素，计算状态
                if(matrix[i][j]=='1'){ // 只有当前位置为1，才可能是正方形的右下角
                    if(i==0 || j==0)    // 第0行第0列，边长为1
                        dp[i][j] = 1;
                    else                // 状态转移求边长
                        dp[i][j] = min(dp[i-1][j-1],min(dp[i-1][j],dp[i][j-1])) + 1; // 状态转移方程
                        // 等价于求dp[i - 1][j]+1，dp[i][j - 1]+1，dp[i - 1][j - 1]+1三者的最小值

                    maxl = max(maxl,dp[i][j]); // 更新最大边长
                }

        return maxl*maxl;
    }
};
// 时间复杂度：**O( mn )**  
// 空间复杂度：**O( mn )**
```

### [300.最长递增子序列](../Coding\动态规划\300.最长递增子序列.md)

```C++
/*
    求最长递增子序列的长度
    dp[i] 为以第 i 个元素结尾的最长上升子序列的长度
    对于每个元素，都遍历之前的dp，看能否接在其后面，可以接就转移试试
*/
class Solution {
public:
    int lengthOfLIS(vector<int>& nums) {
        int n = nums.size();
        vector<int> dp(n,1); // dp[i] 为考虑前 i 个元素，以第 i 个数字结尾的最长上升子序列的长度（序列包含第i个数字）
                            // 初始长度为1，只包含自身

        for(int i=0;i<n;i++){ // 遍历每个下标i
            for(int j=0;j<i;j++){ // 遍历前i个元素
                if(nums[j]<nums[i]) // 如果当前元素nums[i]可以接在nums[j]后面，就更新dp[i]
                    dp[i] = max(dp[i],dp[j]+1);
            }
        }
        return *max_element(dp.begin(),dp.end());
    }
};
// 时间复杂度：**O( n² )**  
// 空间复杂度：**O( n )**

// 动态规划 + 二分：
// 略
// 时间复杂度：**O(n log(n))**  
// 空间复杂度：**O( n )**
```

### [338.比特位计数](../Coding\动态规划\338.比特位计数.md)

```C++
/*
    计算[0,n]中每个数的二进制中1的个数
    res[i]表示i的二进制中1的个数
    状态转移：res[i] = res[i>>1] + (i&1);
*/
class Solution {
public:
    vector<int> countBits(int n) {
        vector<int> res(n+1,0);// res[i]表示i的二进制中1的个数

        for(int i=1;i<=n;i++)
            res[i] = res[i>>1] + (i&1);

        return res;
    }
};
// 时间复杂度：**O(n)**  
// 空间复杂度：**O(1)**
```

### [1277. 统计全为 1 的正方形子矩阵](../Coding\动态规划\1277.-统计全为-1-的正方形子矩阵.md)

```C++
/*
    求只包含 1 的正方形的个数
    dp(i,j) 表示以 (i,j) 为右下角，且只包含 1 的正方形的数量（也是最大正方形的边长）。
*/
class Solution {
public:
    int countSquares(vector<vector<int>>& matrix) {
        int m = matrix.size();
        int n = matrix[0].size();
        vector<vector<int>> dp(m,vector<int>(n,0));// dp(i,j) 表示以 (i,j) 为右下角，且只包含 1 的正方形的数量（也是最大正方形的边长）。
        int ans = 0;

        for(int i=0;i<m;i++)
            for(int j=0;j<n;j++)
                if(matrix[i][j]==1){
                    if(j==0 || i==0)
                        dp[i][j] = 1;
                    else
                        dp[i][j] = min(dp[i-1][j-1],min(dp[i-1][j],dp[i][j-1])) + 1;

                    ans += dp[i][j]; // 不同点：221.最大正方形 这里是求max，而这里是+=
                }

        return ans;
    }
};
// 时间复杂度：**O( mn )**  
// 空间复杂度：**O( mn )**
```

### [剑指-Offer-14--I.-剪绳子](../Coding\动态规划\剑指-Offer-14--I.-剪绳子.md)

```C++
/*
    求长度为n的绳子剪成若干段后的长度最大乘积
    dp[i]表示长度为i的绳子剪完后的最大乘积
    状态转移：从i上剪下j，j*(i-j)表示剪了j后剩下的不剪，j*dp[i-j]表示剪了j后剩下的剪
        dp[i] = max(dp[i], max(j*(i-j), j*dp[i-j]));
*/
class Solution {
public:
    int cuttingRope(int n) {
        vector<int> dp(n+1,0); // dp[i]表示长度为i的绳子剪完后的最大乘积
 
        for (int i = 2; i <= n; i++) { // 枚举每种长度
            for (int j = 1; j < i; j++) // 从i上剪下j
                dp[i] = max(dp[i], max(j*(i-j), j*dp[i-j])); // j*(i-j)表示剪了j后剩下的不剪，j*dp[i-j]表示剪了j后剩下的剪
        }

        return dp[n];
    }
};
// 时间复杂度：**O( n² )**  
// 空间复杂度：**O( n )**
```

### [剑指-Offer-14--II.-剪绳子-II](../Coding\动态规划\剑指-Offer-14--II.-剪绳子-II.md)

```C++
/*
    当 n <= 4 时，可直接返回最大乘积
    当 n > 4 时，可尽可能的拆成多个3，乘积最大，数学证明略
*/
class Solution {
public:
    int cuttingRope(int n) {
        if(n <= 3) return n - 1;
        if(n == 4) return 4;
        long res = 1;
        while(n > 4) 
        {
            n -= 3;
            res *= 3;
            res %= 1000000007;
        }
        // 退出while循环时，最后n的值只有可能是：2、3、4。而2、3、4能得到的最大乘积恰恰就是自身值
        // 因为2、3不需要再剪了（剪了反而变小）；4剪成2x2是最大的，2x2恰巧等于4
        return res * n % 1000000007; 
    }
};
// 时间复杂度：**O( 1 )**  
// 空间复杂度：**O( 1 )**
```

### [剑指-Offer-46.-把数字翻译成字符串](../Coding\动态规划\剑指-Offer-46.-把数字翻译成字符串.md)

```C++
/*
    把数字 (如216612) 翻译成字符串
    数字 0-25 分别对应字母 a-z
    每个数字一定可以单独翻译，也可能和前一个数字一起翻译
*/
class Solution {
public:
    int translateNum(int num) {
        string s = to_string(num);
        // 第一个数字
        int pre = 1,prepre = 1,cur = 1;// pre   表示 以上个字符结尾   的字符串可以翻译的数量
                                       // prepre表示 以上上个字符结尾 的字符串可以翻译的数量
                                       // cur   表示 以当前字符结尾   的字符串可以翻译的数量   
        // 其它数字
        for(int i=1;i<s.size();i++){
            cur = pre; // 当前数字 单独翻译 成字母
            if(s.substr(i-1,2)>="10" && s.substr(i-1,2)<="25") // 当前数字 和前一个数字共同翻译 成字母
                cur += prepre;
            // 准备去处理下一个数字
            prepre = pre;
            pre = cur;
        }

        return cur;
    }
};
// 时间复杂度：**O(n)**  
// 空间复杂度：**O(1)**
```

### [剑指-Offer-47.-礼物的最大价值](../Coding\动态规划\剑指-Offer-47.-礼物的最大价值.md)

```C++
/*
    求从左上角源点到右下角的最大路径和
    dp[i][j]表示从源点到达[i][j]位置能得到的最大礼物价值
    状态转移：左或上的最大值加上当前值
        dp[i][j] = max(dp[i-1][j],dp[i][j-1]) + grid[i][j];
*/
class Solution {
public:
    int maxValue(vector<vector<int>>& grid) {
        int m = grid.size();
        int n = grid[0].size();
        vector<vector<int>> dp(m,vector<int>(n,0)); // dp[i][j]表示从源点到达[i][j]位置能得到的最大礼物价值

        for(int i=0;i<m;i++)
            for(int j=0;j<n;j++){
                if(i==0 && j==0)    // 源点
                    dp[i][j] = grid[i][j];
                else if(i==0)       // 第0行
                    dp[i][j] = dp[i][j-1] + grid[i][j];
                else if(j==0)       // 第0列
                    dp[i][j] = dp[i-1][j] + grid[i][j];
                else
                    dp[i][j] = max(dp[i-1][j],dp[i][j-1]) + grid[i][j];
            }

        return dp[m-1][n-1];
    }
};
// 时间复杂度：**O( mn )**  
// 空间复杂度：**O( mn )**
```

### hard[剑指-Offer-51.-数组中的逆序对](../Coding\动态规划\剑指-Offer-51.-数组中的逆序对.md)

```C++
/*
    下标小但是数值大的两个数构成逆序对
    两个for循环的暴力解时间复杂度是n²，基于归并排序思想降到nlogn
    在归并排序基础上更改带*********号的行即可
*/
class Solution {
public:
    int mergesort(vector<int>& nums,vector<int>& tmp,int L,int R){
        //// 递归
        if(L>=R) // 只有一个元素，逆序对为0
            return 0; // *********
        int mid = (L+R)/2;
        int res = mergesort(nums,tmp,L,mid) + mergesort(nums,tmp,mid+1,R);// 左边的逆序对+右边的逆序对（这里要是mid和mid+1）*********
        
        //// 合并
        int i = L; // 左指针
        int j = mid+1; // 右指针
        int p = L; // 合并指针
        while(i<=mid && j<=R){
            if(nums[i]<=nums[j]){// 这里一定要是小于等于
                tmp[p++] = nums[i++];
                res += (j-(mid+1));// 左指针比右指针小，那么右指针左边的所有元素都比左指针小,*********
            }
            else
                tmp[p++] = nums[j++];
        }
        while(i<=mid){
            tmp[p++] = nums[i++];
            res += (j-(mid+1)); // 左指针没有处理完，那么左指针剩下的每个元素都能和右边的所有元素分别组一个逆序对,*********
        }
        while(j<=R){
            tmp[p++] = nums[j++];
        }
        copy(tmp.begin()+L,tmp.begin()+R+1,nums.begin()+L);

        return res; //*********
    }
    int reversePairs(vector<int>& nums) {
        int n = nums.size();
        vector<int> tmp(n);
        return mergesort(nums,tmp,0,n-1);// *********
    }
};
// 时间复杂度：**O( nlogn )**  
// 空间复杂度：**O( n )**
```

### [剑指-Offer-63.-股票的最大利润](../Coding\动态规划\剑指-Offer-63.-股票的最大利润.md)

```C++
/*
    求只买卖该股票一次可能获得的最大利润是多少
    暴力解法两个for循环时间复杂度n²，空间复杂度为1，用单调栈存当前元素后面的最大元素，时间复杂度降到n，但空间复杂度需要n，滚动变量+动态规划，时间复杂度降到n，空间复杂度为1

    对于每个元素，用滚动变量贪心的记录其前的最小元素，每一天都尝试基于该最小元素卖出股票，贪心的记录最大利润
*/
class Solution {
public:
    int maxProfit(vector<int>& prices) {
        int minprice = INT_MAX; // minprice表示 截止当前，出现过的最小价格
        int maxprof = 0;    // maxprofit表示 在当前及其之前买卖一次能获得的最大利润 

        for(int i:prices){
            maxprof = max(maxprof,i-minprice); // 每一天都看今天卖能不能获得最大利润
            minprice = min(minprice,i); // 贪心记住最小价格
        }

        return maxprof;
    }
};
// 时间复杂度：**O( n )**  
// 空间复杂度：**O( 1 )**
```

## 3、队列 {#customname3}

### [剑指-Offer-59-I.滑动窗口的最大值](../Coding\队列\剑指-Offer-59-I.滑动窗口的最大值.md)

```C++
/*
    利用双端队列构造一个非递增的单调队列，队列首始终放当前窗口中的最大值。
    滑动窗口每滑动一步，考虑窗口左边界划过要丢弃的元素是否是最大值，右边界划过要新增的元素是否可以加入单调队列
*/
class Solution {
public:
    vector<int> maxSlidingWindow(vector<int>& nums, int k) {
        vector<int> res;
        if (nums.size() == 0) return res;

        //// 第一个窗口
        deque<int> Q;
        for (int i = 0; i < k; i++) {
            while (!Q.empty() && Q.back() < nums[i]) // 小于，构造非递增单调队列
                Q.pop_back();
            Q.push_back(nums[i]);
        }
        res.push_back(Q.front()); // 添加窗口最大值

        //// 开始滑动窗口
        for (int i = k; i < nums.size(); i++) {
            // 考虑左边界划过的要丢弃的元素
            if (nums[i-k] == Q.front())
                Q.pop_front();
            // 考虑右边界划过的要新增的元素
            while (!Q.empty() && Q.back() < nums[i])  // 小于，构造非递增单调队列
                Q.pop_back();                      
            Q.push_back(nums[i]);

            res.push_back(Q.front()); // 添加窗口最大值
        }

        return res;
    }
};
// 时间复杂度：**O( n )**  
// 空间复杂度：**O( n )**
```

### [剑指-Offer-59-II.队列的最大值](../Coding\队列\剑指-Offer-59-II.队列的最大值.md)

```C++
/*
    用一个双端队列来实现一个非递增的单调队列，队首元素为当前结算队列中的最大值
*/
class MaxQueue {
public:
    deque<int> tmp; // 双端队列用于实现非递增的单调队列
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

        while(!tmp.empty() && tmp.back() < value) // 小于，构造非递增单调队列
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
// 时间复杂度：**O(n)**  
// 空间复杂度：**O(1)**
```

## 4、二分 {#customname4}

### [35.搜索插入位置](../Coding\二分\35.搜索插入位置.md)

```C++
/*
    找到第一个大于等于target的位置
*/
class Solution {
public:
    int searchInsert(vector<int>& nums, int target) {
        int l =0, r = nums.size()-1;
        int ans = nums.size();// 找大于等于，初始就设为大于所有数，找小于等于，初始就设为小于所有数，这样的话没找到时就直接返回插入位置
        while(l<=r){
            int mid = (l+r)/2;
            if(nums[mid]>=target){// 本质上就是找到第一个大于等于target的位置
                ans = mid;
                r = mid-1;
            }
            else
                l = mid+1;
        }
        return ans;
    }
};
// 时间复杂度：**O( logn )**  
// 空间复杂度：**O( 1 )**
```

### medium[33.搜索旋转排序数组](../Coding\数组\中等\33.搜索旋转排序数组.md)

```C++
/*
    从旋转后的数组中找出`target`值的下标
    直接遍历的时间复杂度是n，二分查找的时间复杂度是logn
    标准的二分基础上，由于数组翻转后变成两个升序部分，L和R之间可能不是有序的，无法直接通过mid和target的相对大小判定应该继续去哪边找，可以先nums[0]和mid比较来判定mid是在哪个升序部分，在左边则[0-mid]必然有序，在右边则[mid,n-1]必然有序
*/
class Solution {
public:
    int search(vector<int>& nums, int target) {
        if(nums.size()==1)
            return nums[0] == target?0:-1;
        int l = 0, r = nums.size()-1;

        while(l<=r){
            int mid = (l+r)/2;
            if(nums[mid]==target)
                return mid;

            // nums数组翻转后变成两个升序部分。
            else if(nums[mid]>=nums[0]){ // 如果nums[0]<=nums[mid]，mid必然在左边的部分
                if(nums[0]<=target && target<nums[mid]) // mid在左边部分，[0-mid]必然有序，可用端点判定
                    r = mid-1;
                else
                    l = mid+1;
            }
            else{                       // 否则mid在右边的部分
                if(nums[mid]<target && target<=nums[nums.size()-1]) // mid在右边部分，[mid,n-1]必然有序，可用端点判断
                    l = mid+1;
                else
                    r = mid-1;
            }
        }
        return -1;
    }
};
// 时间复杂度：**O(logn)**  
// 空间复杂度：**O(1)**
```

### hard[154.寻找旋转排序数组中的最小值-II](../Coding\二分\154.寻找旋转排序数组中的最小值-II.md)

```C++
/*
    从旋转后的数组中找出最小值
    直接遍历的时间复杂度是n，二分查找的时间复杂度是logn
    不同于标准的二分，这里mums[mid]和nums[r]进行比较，因为最小值一定在 r 的左边，因为存在重复元素，比[153. 寻找旋转排序数组中的最小值]多了一个相等的情况
*/
class Solution {
public:
    int minArray(vector<int>& nums) {
        int l = 0, r = nums.size()-1;

        while(l<r){ // 不同于标准二分
            int mid = (r+l)/2;
            if(nums[mid] < nums[r]) // r=mid 的原因是 mid 可能是最小值
                r = mid;    // 不同于标准二分
            else if(nums[mid] > nums[r]) // l=mid+1 的原因是 mid 一定不是最小值
                l = mid+1;
            else
                r--; // 因为相等，一定可以丢弃r，保留mid；([153. 寻找旋转排序数组中的最小值]没有重复元素，不需要这个else)
        }
        return nums[l];
    }
};
// 时间复杂度：**O( logn )**  
// 空间复杂度：**O( 1 )**
```

### [1539.第-k-个缺失的正整数](../Coding\二分\1539.第-k-个缺失的正整数.md)

```C++
/*
    从1开始找到不在数组中的第k个正整数
    直接遍历的时间复杂度是n，二分查找的时间复杂度是logn
    等价于找到最小的下标，这个下标之前缺失k个正整数，判断使用nums[mid]-mid-1 >= k，没有缺则nums[mid]-mid-1等于0，缺1个等于1。

*/
class Solution {
public:
    int findKthPositive(vector<int>& nums, int k) {
        if(nums[0]>k)
            return k; //第一个元素前就缺了k个值，从1开始连续缺，第k个就是k
        int l=0, r=nums.size()-1;

        while(l<=r){
            int mid = (l+r)/2;

            if(nums[mid]-mid-1 >= k) // 没有缺则nums[mid]-mid-1等于0，缺1个等于1
                r = mid-1;
            else 
                l = mid+1;
        } // while退出时l为最小的前面缺失的k个数的下标

        return k-(nums[l-1]-(l-1)-1) + nums[l-1];// nums[l-1] - (l-1)-1为nums[l-1]之前缺了几个，不够k个的加上nums[l-1]就是第k个
    }
};
// 时间复杂度：**O( logn )**  
// 空间复杂度：**O( 1 )**
```

## 5、哈希表 {#customname5}

### [146. LRU 缓存机制](../Coding\哈希表\146.LRU缓存机制.md)

```C++
/*
    对元素按上一次操作的时间排序，最近操作过的元素排在最前，最近最少使用的元素排在最后
    哈希表 + 双向链表
    主要实现get和put函数，unordered_map<int,Node*> mp 用于根据key快速查找链表中的节点
        get函数 
            如果节点不存在，返回-1
            如果节点存在，获取对应节点的值，然后将节点移动到双向链表头部
        put函数  
            如果节点不存在，就创建节点，并将节点放到双向链表的头部，如果容量满了就删除最后的节点；
            如果节点存在，就更新节点，并将节点移动到双向链表的头部
*/
class Node{ // 双向链表的一个节点
public:
    int key,value;
    Node* pre;
    Node* next;
    Node():key(0),value(0),pre(nullptr),next(nullptr){}
    Node(int _key,int _value):key(_key),value(_value),pre(nullptr),next(nullptr){}
};

class LRUCache {
public:
    unordered_map<int,Node*> mp; // 以链表里的每个 节点的值作为键，节点的指针作为值
    int _capacity,size;
    Node* head,*tail;
    LRUCache(int capacity) {
        _capacity = capacity;
        size = 0;
        head = new Node(); // 虚拟节点
        tail = new Node(); // 虚拟节点
        head->next = tail;
        tail->pre = head;
    }
    void addtohead(Node* node){
        node->pre = head;
        node->next = head->next;
        head->next->pre = node;
        head->next = node;
    }
    void remove(Node* node){
        node->pre->next = node->next;
        node->next->pre = node->pre;
    }
    void movetohead(Node* node){ // 关键
        remove(node);
        addtohead(node);
    }
    Node* removetail(){  // 关键
        Node* node = tail->pre;
        remove(node);
        return node;
    }
    
    int get(int key) { // 主要实现
        if(mp.find(key)==mp.end())
            return -1;
        Node* node = mp[key];
        movetohead(node);
        return node->value;
    }
    
    void put(int key, int value) { // 主要实现
        if(mp.find(key)==mp.end()){
            Node* node = new Node(key,value);
            mp[key] = node;
            addtohead(node);
            size++;
            if(size > _capacity){
                Node* node = removetail();
                mp.erase(node->key);
                delete node;
                --size;             // 一开始漏写了这个
            }
        }
        else{
            Node* node = mp[key];
            node->value = value;
            movetohead(node);
        }
    }
};
// time：O(1)
// space：O(capacity)
```

### [705.设计哈希集合](../Coding\哈希表\705.设计哈希集合.md)

```C++
/*
    链地址法：输入的 key % 769 作为哈希集合的 key
    存储：vector<list<int>> data;
*/
class MyHashSet {
public:
    vector<list<int>> data;// 链地址法
    const int base = 769;
    int hash(int key){
        return key % base;
    }
    /** Initialize your data structure here. */
    MyHashSet() :data(769){

    }
    
    void add(int key) {
        int k = hash(key);
        for(auto it = data[k].begin();it!=data[k].end();it++)
            if((*it)==key)
                return;
        data[k].push_back(key);
    }
    
    void remove(int key) {
        int k = hash(key);
        for(auto it = data[k].begin();it!=data[k].end();it++)
            if((*it)==key){
                data[k].erase(it);
                return;             // 这里注意，在范围for循环中删除了元素应该要退出，不然会访问越界
            }
    }
    
    /** Returns true if this set contains the specified element */
    bool contains(int key) {
        int k = hash(key);
        for(auto it = data[k].begin();it!=data[k].end();it++)
            if((*it)==key)
                return true;
        return false;
    }       
};
```

### [706.设计哈希映射](../Coding\哈希表\706.设计哈希映射.md)

```C++
/*
    链地址法：输入的 key % 769 作为哈希表的 key
    存储：vector<list<pair<int,int>>> data;
*/
class MyHashMap {
public:
    vector<list<pair<int,int>>> data;// 链地址法
    static const int base =769;
    static int hash(int key){
        return key%base;
    }
    /** Initialize your data structure here. */
    MyHashMap() :data(base){

    }
    
    /** value will always be non-negative. */
    void put(int key, int value) {
        int k = hash(key);
        for(auto it = data[k].begin();it!=data[k].end();it++)
            if((*it).first == key){
                (*it).second = value;
                return;
            }
        data[k].push_back({key,value});
    }
    
    /** Returns the value to which the specified key is mapped, or -1 if this map contains no mapping for the key */
    int get(int key) {
        int k = hash(key);
        for(auto it = data[k].begin();it!=data[k].end();it++)
            if((*it).first == key)
                return (*it).second;
        return -1;
    }
    
    /** Removes the mapping of the specified value key if this map contains a mapping for the key */
    void remove(int key) {
        int k = hash(key);
        for(auto it = data[k].begin();it!=data[k].end();it++)
            if((*it).first == key){
                data[k].erase(it);
                return;             // 这里注意，在范围for循环中删除了元素应该要退出，不然会访问越界
            }
        return;
    }
};
```

### [1365.有多少小于当前数字的数字](../Coding\哈希表\1365.有多少小于当前数字的数字.md)

```C++
/*
    为每个元素求数组里有多少个元素小于它
    计数排序
    先计数排序，然后动态规划转化为小于等于的个数
*/
class Solution {
public:
    vector<int> smallerNumbersThanCurrent(vector<int>& nums) {
        vector<int> cnt(101,0);// 以nums的值为下标
        // 计数排序
        for(int i:nums)
            cnt[i]++;
        // 转化为小于等于的个数
        for(int i=1;i<101;i++)
            cnt[i] += cnt[i-1];

        vector<int> res;
        for(int i:nums)// 构造答案
            res.push_back(i==0 ? 0 : cnt[i-1]);// 注意这里的i==0的情况，小于等于i-1，就是小于i

        return res;
    }
};
// 时间复杂度：**O( n )**  
// 空间复杂度：**O( n )**
```

### [1512.好数对的数目](../Coding\哈希表\1512.好数对的数目.md)

```C++
/*
    下标不等 但 值相等 的两个元素构成好数对
    统计频数，然后求和每种频数的Cn2 ( n*(n-1)/2 )
*/
class Solution {
public:
    int numIdenticalPairs(vector<int>& nums) {
        // 统计频数
        unordered_map<int,int> mp;
        for(int num:nums)
            ++mp[num];
        // 求和 每种频数的Cn2（排列组合）
        int ans = 0;
        for(auto p: mp)
            ans += p.second*(p.second-1)/2; // 这里实际上就是求Cn2，n*(n-1)/2

        return ans;
    }
};
// 时间复杂度：**O(n)**  
// 空间复杂度：**O(1)**
```

### [剑指Offer-03.数组中重复的数字](../Coding\哈希表\剑指Offer-03.数组中重复的数字.md)

```C++
/*
    使用哈希表需要额外n的空间复杂度，原地置换的方式空间复杂度为1
    原地置换：  
    下标是 `[0,n-1]`，元素也是 `[0,n-1]`，
        如果不存在重复元素，所有元素要么下标和元素相等，要么可以通过一条链：`i -> nums[i] -> nums[nums[i]]`串起来，
        如果出现了重复元素，那么必然存在环，不断地交换 `swap(nums[i],nums[nums[i]])`，必然会回到自身`nums[i] == nums[nums[i]]`
*/
class Solution {
public:
    int findRepeatNumber(vector<int>& nums) {
        // 从第一个元素开始换，换到和下标相等，然后再第二个元素
        for(int i=0;i<nums.size();i++)
            while(nums[i]!=i) { // 只要下标和元素不等就去不停的交换
                if(nums[i] == nums[nums[i]]) // 如果找到重复元素
                    return nums[i];
                swap(nums[i],nums[nums[i]]);
            }
        return -1;
    }
};
// 时间复杂度：**O(n)**  
// 空间复杂度：**O(1)**
```

### [面试题-01.04.-回文排列](../Coding\哈希表\面试题-01.04.-回文排列.md)

```C++
/*
    判断字符串是否是回文串的排列之一
    基于哈希集合，每个元素不在集合中就加入，在集合中就删除，如果是回文排列，最后集合size应该小于等于1
*/
class Solution {
public:
    bool canPermutePalindrome(string s) {
        unordered_set<char> set;
        for(char c:s) // 没有就插进去，已经有了就删除
            if(set.find(c)!=set.end())
                set.erase(c);
            else 
                set.insert(c);

        return set.size() <= 1; // 如果是回文，最后最多只能剩下一个
    }
};
// 时间复杂度：**O(n)**  
// 空间复杂度：**O(1)**
```

## 6、回溯 {#customname6}

### [二进制手表](../Coding\回溯\401.二进制手表.md)

```C++
/*
    求二进制手部点亮turnedOn个灯，可以得到多少种时间
    回溯
*/
class Solution {
public:
    vector<string> res; // 用于存储结果
    unordered_map<int,int> mp{{0,8},{1,4},{2,2},{3,1},{4,32},{5,16},{6,8},{7,4},{8,2},{9,1}}; // 通过哈希表将灯的下标和数字联系起来
    void backward(int num,int start,pair<int,int>& time){ // 从下标start开始找num个灯点亮
        if(num==0) { // 如果灯找够了，就表示得到一个排列组合

            // 如果这个排列组合的时间不满足要求，就不保存
            if(time.first>11 || time.second>59)
                return;

            // 时间满足要求，保存
            string hour = to_string(time.first);
            string second = to_string(time.second);
            if(second.size()==1) // 分钟前如果需要增加0
                second.insert(0,"0");
            res.push_back(hour+":"+second);
            return;
        }
        // 从start下标开始继续去点亮
        for(int i=start;i<10;i++) {
            pair<int,int> tmp = time;
            // 点亮第i个灯 
            if(i<4)
                time.first += mp[i];
            else    
                time.second += mp[i];
            backward(num-1,i+1,time);
            // 不点亮第i个灯
            time = tmp;
        }
    }

    vector<string> readBinaryWatch(int turnedOn) {
        pair<int,int> time{0,0}; // 回溯时用于存储当前的时间，first为小时，second为分钟
        backward(turnedOn,0,time);// 从第0位开始，挑出turnedOn位，如果turnedOn位表示的时间time满足要求，就添加到res

        return res;
    }
};
```

## 7、链表 {#customname7}

### [19.删除链表的倒数第N个节点](../Coding\链表\19.删除链表的倒数第N个节点.md)

```C++
/*
    双指针：
        开始快慢指针都指向哨兵节点，
        然后快指针先走n+1步，再快慢指针一起走，快指针走到nullptr时，慢指针走到倒数第n+1个节点，
        然后 `slow->next = slow->next->next` 删除倒数第n个节点
*/
struct ListNode {
    int val;
    ListNode *next;
    ListNode() : val(0), next(nullptr) {}
    ListNode(int x) : val(x), next(nullptr) {}
    ListNode(int x, ListNode *next) : val(x), next(next) {}
};

class Solution {
public:
    ListNode* removeNthFromEnd(ListNode* head, int n) {
        ListNode* tmp = new ListNode();
        tmp->next = head;
        ListNode* slow = tmp;
        ListNode* fast = tmp;

        for(int i=1;i<=n+1;i++)// 循环n+1次
            fast = fast->next;
        while(fast!=nullptr){
            slow = slow->next;
            fast = fast->next;
        }

        slow->next = slow->next->next;

        return tmp->next;
    }
};
// 时间复杂度：**O(n)**  
// 空间复杂度：**O(1)**
```

### [23.合并K个升序链表](../Coding\链表\23.合并K个升序链表.md)

```C++
/*
    合并两个有序链表的思路基础上，增加使用优先级队列
*/
struct ListNode {
    int val;
    ListNode *next;
    ListNode() : val(0), next(nullptr) {}
    ListNode(int x) : val(x), next(nullptr) {}
    ListNode(int x, ListNode *next) : val(x), next(next) {}
};
class Node1{
public:
    int val;
    ListNode* ptr;
    Node1(){}
    Node1(int _val,ListNode* _ptr):val(_val),ptr(_ptr){}
    Node1(int _val):val(_val),ptr(nullptr){}
    bool operator<(const Node& b) const{ // 使用标准模板库的优先级队列要注意必须重载  <  运算符，返回true表示需要调整。
        return this->val > b.val; //小于是最大堆，大于是最小堆
    }
};
class Solution {
public:
    priority_queue<Node1> q;// 默认是最大堆
    ListNode* mergeKLists(vector<ListNode*>& lists) {
        if(lists.size()==0)
            return nullptr;
        // 第一个节点入堆
        for(ListNode* node:lists)
            if(node!=nullptr)
                q.push(Node1(node->val,node));
        // 合并链表
        ListNode* tmp = new ListNode(); // 虚拟节点
        ListNode* res = tmp;
        while(!q.empty()) { // 合并
            Node1 p = q.top();
            q.pop();
            res->next = p.ptr;
            res = res->next;

            if(p.ptr->next!=nullptr)
                q.push(Node1(p.ptr->next->val,p.ptr->next));
        }

        return tmp->next;
    }
};
// time：O(kn x logk)
// space：O(k)
```

### [25.-K-个一组翻转链表](../Coding\链表\25.-K-个一组翻转链表.md)

```C++
/*
    构造四个指针，位置关系：pre，head，...... ，tail，nex。
    head 到 tail 有 k 个节点，翻转这 k 个节点的链表，然后四个指针都往后移动 k+1 步，翻转后续的 k 个节点
*/
struct ListNode {
    int val;
    ListNode *next;
    ListNode() : val(0), next(nullptr) {}
    ListNode(int x) : val(x), next(nullptr) {}
    ListNode(int x, ListNode *next) : val(x), next(next) {}
};

class Solution {
public:
    pair<ListNode*, ListNode*> myReverse(ListNode* head, ListNode* tail) { // 翻转一个子链表，并且返回新的头与尾
        // 三个指针实现翻转：prev，cur，head_
        ListNode* prev = nullptr;
        ListNode* cur = nullptr;
        ListNode* head_ = head;
        while (prev != tail) {
            cur = head_;
            head_ = head_->next;

            cur->next = prev;
            prev = cur;
        }
        return {tail, head};
    }

    ListNode* reverseKGroup(ListNode* head, int k) {
        ListNode* hair = new ListNode(0); // 虚拟头节点
        hair->next = head;
        ListNode* pre = hair;

        // 四个指针的位置关系：pre，head，...... ，tail，nex
        while (head) {
            ListNode* tail = pre;
            
            // 获取待翻转的尾部节点，同时查看剩余部分长度是否大于等于 k
            for (int i = 0; i < k; ++i) {
                tail = tail->next;
                if (!tail) {
                    return hair->next;
                }
            }
            ListNode* nex = tail->next;
            pair<ListNode*, ListNode*> result = myReverse(head, tail);
            head = result.first;
            tail = result.second;
            
            // 把子链表重新接回原链表
            pre->next = head;
            tail->next = nex;
            // 移动指针准备下一次翻转
            pre = tail;
            head = tail->next;
        }

        return hair->next;
    }
};
```

### [82.删除链表中的重复元素](../Coding\链表\82.删除链表中的重复元素.md)

```C++
/*
    删除一个已排序的链表里所有重复的元素（有三个A，则三个A全删除）
    设置一个哨兵节点，如果哨兵节点的后两个节点值重复，就不断删除哨兵节点后的重复节点，直到哨兵节点的后两个节点的值不重复，此时哨兵节点后一步。
*/
struct ListNode {
    int val;
    ListNode *next;
    ListNode() : val(0), next(nullptr) {}
    ListNode(int x) : val(x), next(nullptr) {}
    ListNode(int x, ListNode *next) : val(x), next(next) {}
};

class Solution {
public:
    ListNode* deleteDuplicates(ListNode* head) {
        // 虚拟头结点
        ListNode* A = new ListNode();
        A->next = head;
        ListNode* tmp = A; // 哨兵节点

        while(tmp->next!=nullptr && tmp->next->next!=nullptr){ // 哨兵节点后两个节点不为空
            if(tmp->next->val == tmp->next->next->val){ // 如果哨兵节点后两个节点值相等
                int t = tmp->next->val;
                while(tmp->next!=nullptr && tmp->next->val==t) // 不断删除哨兵节点后的重复节点
                    tmp->next = tmp->next->next; // 删除哨兵节点的后一个节点
            }
            else // 如果哨兵节点后两个节点值不相等，后移哨兵节点
                tmp = tmp->next;
        }

        return A->next;
    }
};
// 时间复杂度：**O(n)**  
// 空间复杂度：**O(1)**
```

### [141.环形链表-I](../Coding\链表\141.环形链表-I.md)

```C++
/*
    判断链表是否有环
    双指针
    快慢指针同时指向头节点，然后快指针走两步，慢指针走一步，如果快慢指针会相等则有环，不会相等则没有环
*/
struct ListNode {
    int val;
    ListNode *next;
    ListNode() : val(0), next(nullptr) {}
    ListNode(int x) : val(x), next(nullptr) {}
    ListNode(int x, ListNode *next) : val(x), next(next) {}
};

class Solution141 {
public:
    bool hasCycle(ListNode *head) {
        ListNode *fast,*slow;
        fast = slow = head;

        while(fast && fast->next) {
            slow = slow->next;
            fast = fast->next->next;

            if(slow == fast) {
                return true;
            }
        }

        return false;
    }
};
// 时间复杂度：**O(n)**  
// 空间复杂度：**O(1)**
```

### [142.环形链表-II](../Coding\链表\142.环形链表-II.md)

```C++
/*
    判断链表是否有环，如果有，返回入环点
    双指针
    快慢指针同时指向头节点，然后快指针走两步，慢指针走一步，如果快慢指针会相等则有环，不会相等则没有环
    有环时，快慢指针的相遇点到入环点的距离加上 n-1 圈的环长，恰好等于从链表头部到入环点的距离
    因此快慢指针相遇时，再用一个指针从头开始，其和慢指针的相遇点就是入环点
*/
struct ListNode {
    int val;
    ListNode *next;
    ListNode() : val(0), next(nullptr) {}
    ListNode(int x) : val(x), next(nullptr) {}
    ListNode(int x, ListNode *next) : val(x), next(next) {}
};

class Solution142 {
public:
    ListNode *detectCycle(ListNode *head) {
        ListNode *fast,*slow;
        fast = slow = head;

        while (fast != nullptr && fast->next !=nullptr) {// 注意这个判断条件
            slow = slow->next;
            fast = fast->next->next;
            
            if (fast == slow) { // 如果快慢指针相遇，说明有环
                ListNode *ptr = head;
                while (ptr != slow) {   // ptr和slow一起走，相遇点就是入环点
                    ptr = ptr->next;
                    slow = slow->next;
                }
                return ptr;    // 返回入环点
            }
        }

        return nullptr;
    }
};
// 时间复杂度：**O(n)**  
// 空间复杂度：**O(1)**
```

### [148.排序链表](../Coding\链表\148.排序链表.md)

```C++
/*
    归并排序
    sort函数执行两步：
    1、快慢指针找到中间节点，以中间节点分成两个链表head1和head2
    2、Merge(Sort(head1),Sort(head2))，其中merge函数合并两个有序链表
*/
struct ListNode {
    int val;
    ListNode *next;
    ListNode() : val(0), next(nullptr) {}
    ListNode(int x) : val(x), next(nullptr) {}
    ListNode(int x, ListNode *next) : val(x), next(next) {}
};

class Solution {
public:
    ListNode* Merge(ListNode* list1,ListNode* List2){ // 合并两个排序链表
        ListNode* head = new ListNode(); // 虚拟节点
        ListNode* tmp = head;

        while(list1!=nullptr && List2!=nullptr){
            if(list1->val < List2->val){
                tmp->next = list1;
                list1 = list1->next;
            }   
            else{
                tmp->next = List2;
                List2 = List2->next;
            }
            tmp = tmp->next;
        }
        if(list1!=nullptr)
            tmp->next = list1;
        if(List2!=nullptr)
            tmp->next = List2;

        return head->next;
    }
    ListNode* Sort(ListNode* head){
        ListNode* slow = head;
        ListNode* fast = head->next;// 这个fast的初始值很关键
        if(fast==nullptr)// 只有一个节点时
            return head;
        while(fast!=nullptr && fast->next!=nullptr){// 快慢指针找到中间节点
            slow = slow->next;
            fast = fast->next->next;
        }
        ListNode* head1 = head;
        ListNode* head2 = slow->next;
        slow->next = nullptr;   // 拆成两个链表
        return Merge(Sort(head1),Sort(head2));// 这个递归写法很好
    }
    ListNode* sortList(ListNode* head) {
        if(head==nullptr)
            return head;
        return Sort(head);
    }
};
// time：O(nlogn)
// space：O(n)
```

### [160.相交链表](../Coding\链表\160.相交链表.md)

```C++
/*
    求两个相交链表的交点
    双指针：
        两个指针分别从两个链表头出发，走到链表尾部跳转到另一个链表头继续走，最终在相交节点相遇
        两个指针最终必然同时到达最后一个节点，因为此时两个指针走过的路程都是两个链表的长度和。同时到达最后一个节点，必然也会同时到达相交节点
*/
struct ListNode {
    int val;
    ListNode *next;
    ListNode() : val(0), next(nullptr) {}
    ListNode(int x) : val(x), next(nullptr) {}
    ListNode(int x, ListNode *next) : val(x), next(next) {}
};

class Solution160 {
public:
    ListNode *getIntersectionNode(ListNode *headA, ListNode *headB) {
        ListNode* p1 = headA;
        ListNode* p2 = headB;

        while(p1!=p2){// 不相交时p1和p2必然同时等于nullptr，此时都走完两个链表长度，然后退出while
            p1 = p1 == nullptr? headB:p1->next;
            p2 = p2 == nullptr? headA:p2->next;
        }

        return p1;
    }
};
// 时间复杂度：**O(m+n)**  
// 空间复杂度：**O(1)**
```

### [206.反转链表](../Coding\链表\206.反转链表.md)

```C++
/*
    迭代解法：
    三个指针，一个head指向还没有被反转的一部分，cur指向准备要反转的那个节点，pre指向已经反转了的一部分，反转操作:
        `cur = head;`  
        `head = head->next;`  
        `cur->next = pre;`  
        `pre = cur;`
*/
struct ListNode {
    int val;
    ListNode *next;
    ListNode() : val(0), next(nullptr) {}
    ListNode(int x) : val(x), next(nullptr) {}
    ListNode(int x, ListNode *next) : val(x), next(next) {}
};
class Solution {
public:
    ListNode* reverseList(ListNode* head) {
        if(head==nullptr || head->next==nullptr)
            return head;
        ListNode* pre=nullptr,*cur=nullptr;

        while(head!=nullptr){
            cur = head;
            head = head->next;
            cur->next = pre;
            pre = cur;
        }

        return pre;
    }
};
// 时间复杂度：**O(n)**  
// 空间复杂度：**O(1)**
```

### [234.回文链表](../Coding\链表\234.回文链表.md)

```C++
/*
    快慢指针找到链表中点的同时翻转前半部分链表，然后从中间开始判断回文：
    快慢指针的同时，用slow指针作为head，结合cur、pre指针反转前半部分链表，
    然后cur和slow同时走，如果一直相等，则是回文链表
*/
struct ListNode {
    int val;
    ListNode *next;
    ListNode() : val(0), next(nullptr) {}
    ListNode(int x) : val(x), next(nullptr) {}
    ListNode(int x, ListNode *next) : val(x), next(next) {}
};

class Solution234 {
public:
    bool isPalindrome(ListNode* head) {
        if(!head || !head->next)
            return 1;
        // 快慢指针找中间节点，同时翻转前半部分
        ListNode* fast=head,*slow=head;
        ListNode* pre = nullptr,*cur = nullptr;
        while(fast && fast->next) {
            cur = slow;
            slow = slow->next;
            fast = fast->next->next;
            
            cur->next = pre;
            pre = cur;
        }
        if(fast) // 链表奇数长度就跳过一个
            slow = slow->next;
        // 判断回文
        while(cur) {
            if(cur->val != slow->val)
                return 0;
            cur = cur->next;
            slow = slow->next;
        }

        return 1;
    }
};
// 时间复杂度：**O(n)**  
// 空间复杂度：**O(1)**
```

### [876.链表的中间结点](../Coding\链表\876.链表的中间结点.md)

```C++
/*
    快慢指针，返回slow即是中间节点
*/
struct ListNode {
    int val;
    ListNode *next;
    ListNode() : val(0), next(nullptr) {}
    ListNode(int x) : val(x), next(nullptr) {}
    ListNode(int x, ListNode *next) : val(x), next(next) {}
};

class Solution876 {
public:
    ListNode* middleNode(ListNode* head) {
        ListNode* slow = head;
        ListNode* fast = head;

        while (fast != NULL && fast->next != NULL) {
            slow = slow->next;
            fast = fast->next->next;
        }

        return slow;
    }
};
// 时间复杂度：**O(n)**  
// 空间复杂度：**O(1)**
```

### [1290.二进制链表转整数](../Coding\链表\1290.二进制链表转整数.md)

```C++
/*
    遍历一遍链表；二进制从高位起，计算十进制值的方法为：  
        `ans = ans * 2 + cur->val;`
*/
struct ListNode {
    int val;
    ListNode *next;
    ListNode() : val(0), next(nullptr) {}
    ListNode(int x) : val(x), next(nullptr) {}
    ListNode(int x, ListNode *next) : val(x), next(next) {}
};

class Solution1290 {
public:
    int getDecimalValue(ListNode* head) {
        ListNode* cur = head;
        int ans = 0;

        while (cur != nullptr) {
            ans = ans * 2 + cur->val;
            cur = cur->next;
        }

        return ans;
    }
};
// 时间复杂度：**O(n)**  
// 空间复杂度：**O(1)**
```

### [剑指Offer-25.合并两个排序的链表](../Coding\链表\剑指Offer-25.合并两个排序的链表.md)

```C++
/*
    双指针：
        两个指针分别指向两个链表，然后不断的比较值大小，小的拿出来排序，然后小的指针往后移
*/
struct ListNode {
    int val;
    ListNode *next;
    ListNode() : val(0), next(nullptr) {}
    ListNode(int x) : val(x), next(nullptr) {}
    ListNode(int x, ListNode *next) : val(x), next(next) {}
};

class Solution {
public:
    ListNode* mergeTwoLists(ListNode* l1, ListNode* l2) { // 合并两个有序链表
        ListNode* head = new ListNode(); // 虚拟头结点
        ListNode* tmp = head;
        // 使用双指针的形式合并
        while(l1!=nullptr && l2!=nullptr){
            if(l1->val < l2->val){
                tmp->next = l1;
                l1 = l1->next;
            }
            else{
                tmp ->next = l2;
                l2 = l2->next;
            }
            tmp = tmp->next;
        }
        // 将某个链表剩下的节点直接接到结果尾部
        if(l1!=nullptr)
            tmp->next = l1;
        if(l2!=nullptr)
            tmp->next = l2;

        return head->next;
    }
};
// time：O(n+m)
// space：O(1)
```

### [剑指Offer-35.复杂链表的复制](../Coding\链表\剑指Offer-35.复杂链表的复制.md)

```C++
/*
    递归解法：
        复制当前节点，然后分别调用递归复制next节点和random节点，  
        递归终止条件有两个：如果要复制的节点为nullptr，就返回nullptr，如果不为nullptr且已经复制过了，返回mp[head]  
        ```unordered_map<Node*,Node*> mp; // 存储已经完成复制的节点对```

    迭代解法：
        在原始链表的每个节点后都复制和插入前一个节点，总的节点个数翻倍，  
        然后复制random节点：```ptr->next->random = (ptr->random==nullptr)?nullptr:ptr->random->next;```  
        然后将链表按奇数和偶数拆成两个，即完成链表的复制。
*/
class Node {
public:
    int val;
    Node* next;
    Node* random;
    
    Node(int _val) {
        val = _val;
        next = NULL;
        random = NULL;
    }
};
class Solution {               // 递归解法
public:
    unordered_map<Node*,Node*> mp; // 存储已经完成复制的节点对
    Node* copyRandomList(Node* head) {
        //// 递归终止条件
        if(head==nullptr)
            return head;
        if(mp.find(head)!=mp.end()) // 如果已经复制过了
            return mp[head];
        //// 复制节点
        Node* node = new Node(head->val);
        mp[head] = node;
        node->next = copyRandomList(head->next);
        node->random = copyRandomList(head->random);

        return node;
    }
};
class Solution {            // 迭代解法
public:
    Node* copyRandomList(Node* head) {
        if(head==nullptr)
            return head;
        // 复制节点
        Node* ptr = head;
        while(ptr!=nullptr){
            Node* tmp = new Node(ptr->val);
            tmp->next = ptr->next;
            ptr->next = tmp;

            ptr = ptr->next->next;
        }
        // 复制random
        ptr = head;
        while(ptr!=nullptr){
            ptr->next->random = (ptr->random==nullptr)?nullptr:ptr->random->next;

            ptr = ptr->next->next;
        }
        // 拆分
        Node* oldlist = head;
        Node* newlist = head->next;
        Node* tmp = head->next;
        while(oldlist!=nullptr){
            oldlist->next = oldlist->next->next;
            newlist->next = (newlist->next==nullptr)?nullptr:newlist->next->next;

            oldlist = oldlist->next;
            newlist = newlist->next;
        }

        return tmp;
    }
};
```

## 8、排序 {#customname8}

### [15.三数之和](../Coding\排序\15.三数之和.md)

```C++
/*
    直接三重for循环的时间复杂度是n³，排序+双指针 可降为n²
*/
class Solution {
public:
    void quicksort(vector<int>& nums,int L,int R) {
        if(L>=R)
            return;
        int p = rand()%(R-L+1)+L;
        swap(nums[p],nums[R]);
        int i=L-1;
        for(int j=L;j<R;j++)
            if(nums[j]<=nums[R])
                swap(nums[j],nums[++i]);
        swap(nums[++i],nums[R]);
        quicksort(nums,L,i-1);
        quicksort(nums,i+1,R);
    }
    vector<vector<int>> threeSum(vector<int>& nums) {
        int n = nums.size();
        vector<vector<int>> res;

        // srand((unsigned)time(nullptr));
        quicksort(nums,0,n-1);// 排序

        for(int first=0;first<n;first++){// 枚举第一个元素
            if(first>0 && nums[first] == nums[first-1])// 避免重复
                continue;
            int target = -nums[first];// 双指针求第二个元素和第三个元素的和的目标值
            int second = first+1; // 头指针
            int third = n-1;      // 尾指针
            while(second < third) {
                if (second>first+1 && nums[second] == nums[second-1]) // 避免重复
                    second++;
                else if (nums[second]+nums[third] > target) // 三者之和大于0，尾指针左移
                    third--;
                else if (nums[second]+nums[third] < target) // 三者之和小于0，头指针右移
                    second++;
                else {                                    // 三者之和等于0
                    res.push_back({nums[first],nums[second],nums[third]});
                    second++;
                }
            }
        }

        return res;
    }
};
// 时间复杂度：**O(n²)**  
// 空间复杂度：**O(logn)**
```

### [1122.数组的相对排序](../Coding\排序\1122.数组的相对排序.md)

```C++
/*
    快速排序
*/              
class Solution {
public:
    unordered_map<int,int> mp;
    void quicksort(vector<int>& nums,int L,int R){
        if(L>=R)
            return;
        int p = rand()%(R-L+1)+L;
        swap(nums[R],nums[p]);
        int i = L-1;
        for(int j = L; j <= R-1; j++)
            if(mp[nums[j]] < mp[nums[R]] || (mp[nums[j]] == mp[nums[R]] && nums[j] <= nums[R]))// 注意判断条件
                swap(nums[j], nums[++i]);
        swap(nums[++i], nums[R]);
        quicksort(nums, L, i-1);
        quicksort(nums, i+1, R);
    }
    vector<int> relativeSortArray(vector<int>& arr1, vector<int>& arr2) {
        for(int i=0;i<arr2.size();i++) // 放arr2到mp
            mp[arr2[i]] = i;
        for(int i=0;i<arr1.size();i++) // 放arr1比arr2多的那些元素到mp
            if(mp.find(arr1[i])==mp.end())
                mp[arr1[i]] = 2000;
        // srand((unsigned)time(nullptr));
        quicksort(arr1,0,arr1.size()-1);

        return arr1;
    }
};
// 时间复杂度：**O(nlogn)**  
// 空间复杂度：**O(logn)**
```

### [1636.按照频率将数组升序排序](../Coding\排序\1636.按照频率将数组升序排序.md)

```C++
/*
    按频率升序，频率相同，按值降序排序
    快速排序
*/
class Solution {
public:
    unordered_map<int,int> mp;
    bool less_(int a,int b){
        return mp[a] < mp[b] || (mp[a]==mp[b] && a>b); // 注意判断条件
    }
    void quicksort(vector<int>& nums,int L,int R){
        if(L>=R)
            return;
        int p = rand()%(R-L+1)+L;
        swap(nums[R],nums[p]);
        int i = L-1;
        for(int j=L;j<R;j++)
            if(less_(nums[j],nums[R]))
                swap(nums[j],nums[++i]);
        swap(nums[++i],nums[R]);
        quicksort(nums, L, i-1);
        quicksort(nums, i+1, R);
    }
    vector<int> frequencySort(vector<int>& nums) {
        for(int i:nums)
            mp[i]++;
        // srand((unsigned)time(nullptr));
        quicksort(nums,0,nums.size()-1);

        return nums;
    }
};
// 时间复杂度：**O(nlogn)**  
// 空间复杂度：**O(logn)**
```

### [剑指Offer-45.把数组排成最小的数](../Coding\排序\剑指Offer-45.把数组排成最小的数.md)

```C++
/*
    快速排序
*/
class Solution {
public:
    bool less_(int a,int b){
        return to_string(a)+to_string(b) < to_string(b)+to_string(a); // 如果a放在b的前面更小，就把a放在b的前面，注意这个的传递性是需要证明的
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
        quicksort(nums, L, i-1);
        quicksort(nums, i+1, R);
    }
    string minNumber(vector<int>& nums) {
        // srand((unsigned)time(nullptr));
        quicksort(nums,0,nums.size()-1);

        string res ="";
        for(int i:nums)
            res += to_string(i);

        return res;
    }
};
// 时间复杂度：**O(nlogn)**  
// 空间复杂度：**O(logn)**
```

### [剑指Offer-61.扑克牌中的顺子](../Coding\排序\剑指Offer-61.扑克牌中的顺子.md)

```C++
/*
    判断5张扑克牌是否能构成顺子
*/
class Solution {
public:
    bool isStraight(vector<int>& nums) {
        sort(nums.begin(),nums.end());// 数组排序

        int joker = 0; // 记录大小王数量
        for(int i=0;i<4;++i){ // 看前4张牌
            if(nums[i] == 0)
                joker++; // 统计大小王数量
            else if(nums[i] == nums[i+1])
                return false; // 若有重复的非大小王，返回 false
        }

        return nums[4] - nums[joker] < 5; // 最大牌 - 最小牌 < 5 则可构成顺子
    }
};
// 时间复杂度：**O(n)**  
// 空间复杂度：**O(1)**
```

## 9、树 {#customname9}

### [94.二叉树的中序遍历](../Coding\树\94.二叉树的中序遍历.md)

```C++
/*
    
*/
struct TreeNode {
    int val;
    TreeNode *left;
    TreeNode *right;
    TreeNode(int x) : val(x), left(NULL), right(NULL) {}
};

class Solution { // 递归解法
public:
    void order(TreeNode* root, vector<int>& res) {
        if(!root)// 如果root为空
            return ;
        // 先序遍历
        // res.push_back(root->val);
        // order(root->left,res);
        // order(root->right,res);

        // 中序遍历
        order(root->left,res);
        res.push_back(root->val);
        order(root->right,res);

        // 后序遍历
        // order(root->left,res);
        // order(root->right,res);
        // res.push_back(root->val);
    }
    vector<int> orderTraversal(TreeNode* root) {
        vector<int> res;
        order(root,res);
        return res;
    }
};
```

### [100.相同的树](../Coding\树\100.相同的树.md)

```C++
/*
*/
struct TreeNode {
    int val;
    TreeNode *left;
    TreeNode *right;
    TreeNode(int x) : val(x), left(NULL), right(NULL) {}
};

class Solution {
public:
    bool isSameTree(TreeNode* p, TreeNode* q) {
        if(p && q && p->val==q->val)
            return isSameTree(p->left,q->left) && isSameTree(p->right,q->right);
        else if(!p && !q)
            return true;
        else 
            return false;
    }
};
```

### [101.对称二叉树](../Coding\树\101.对称二叉树.md)

```C++
/*
*/
struct TreeNode {
    int val;
    TreeNode *left;
    TreeNode *right;
    TreeNode(int x) : val(x), left(NULL), right(NULL) {}
};

class Solution {
public:
    bool dfs(TreeNode*root1,TreeNode*root2){ // 判断 root1子树 和 root2子树 是否轴对称的
        if(root1 && root2 && root1->val==root2->val)
            return dfs(root1->left,root2->right) && dfs(root1->right,root2->left);
        else if(!root1 && !root2)
            return true;
        else 
            return false;
    }
    bool isSymmetric(TreeNode* root) {
        return dfs(root->left,root->right);
    }
};
```

### [102.二叉树的层序遍历](../Coding\树\102.二叉树的层序遍历.md)

```C++
/*
*/
struct TreeNode {
    int val;
    TreeNode *left;
    TreeNode *right;
    TreeNode(int x) : val(x), left(NULL), right(NULL) {}
};

// 递归
class Solution {
public:
    void leveldfs(TreeNode*root,int level,vector<vector<int>>& res){
        if(root==nullptr)
            return;
        if(level>=res.size()) // 如果是没有访问过的新层
            res.push_back(vector<int>());

        res[level].push_back(root->val);
        leveldfs(root->left,level+1,res);
        leveldfs(root->right,level+1,res);
    }
    vector<vector<int>> levelOrder(TreeNode* root) {
        vector<vector<int>> res; // 存储层序遍历的结果
        leveldfs(root,0,res);
        return res;
    }
};
// 非递归
class Solution {
public:
    vector<vector<int>> levelOrder(TreeNode* root) {
        vector<vector<int>> res;
        if(root == nullptr)
            return res;

        queue<TreeNode*> Q;
        TreeNode* T = root;
        Q.push(T);
        while(!Q.empty()){ // 一次while就是遍历一层
            int n = Q.size(); // 当前层的节点个数
            vector<int> tmp; // 添加用于存储当前层节点的新数组
            for(int i=0;i<n;i++){ // 访问当前层的所有节点，然后将他们的子结点加到队列后面
                T = Q.front();Q.pop();
                tmp.push_back(T->val);
                if(T->left != nullptr)
                    Q.push(T->left);
                if(T->right != nullptr)
                    Q.push(T->right);
            }
            res.push_back(tmp);
        }

        return res;
    }
};
```

### [104.二叉树的最大深度](../Coding\树\104.二叉树的最大深度.md)

```C++
/*
*/
struct TreeNode {
    int val;
    TreeNode *left;
    TreeNode *right;
    TreeNode(int x) : val(x), left(NULL), right(NULL) {}
};
// 广度优先
class Solution {
public:
    int maxDepth(TreeNode* root) {
        int depth=0;
        if(root == nullptr)
            return depth;

        queue<TreeNode*> Q;
        TreeNode *T = root;
        Q.push(T);
        while(!Q.empty()){
            int len = Q.size(); // 当前层的结点数
            for(int i=0;i<len;i++){ // 遍历当前层的所有结点
                TreeNode* T = Q.front();Q.pop();
                if(T->left)
                    Q.push(T->left);
                if(T->right)
                    Q.push(T->right);
            }
            depth++; //遍历当前层的所有结点后，层数depth++
        }

        return depth;
    }
};
// 深度优先
class Solution {
public:
    int maxDepth(TreeNode* root) {
        if(root == nullptr)
            return 0;

        int l = maxDepth(root->left);
        int r = maxDepth(root->right);
        return max(l,r)+1;
    }
};
```

## 10、数学 {#customname10}

### [69.-x-的平方根](../Coding\数学\69.-x-的平方根.md)

```C++
/*
    二分查找
*/
class Solution {
public:
    int mySqrt(int x) {
        int l=0, r=x;
        int ans=0; // 这里ans的初始值可设为最小值0

        while(l<=r){
            int mid = (r+l)/2;
            if((long long)mid*mid <= x){
                ans = mid;
                l = mid+1;
            }
            else
                r = mid-1;
        }

        return ans;
    }
};
// 时间复杂度：**O(logn)**  
// 空间复杂度：**O(1)**
```

### [172.阶乘后的零](../Coding\数学\172.阶乘后的零.md)

```C++
/*
    返回 n! 结果中尾随零的数量
    核心思想就是统计阶乘的连乘式中5的个数，(因为2的倍数比5的倍数多，0的个数由5的个数决定)，注意5的倍数都可以拆成5和其他的一个整数的乘积。`最终 5 的个数就是 n/5 + n/25 + n/125 + ...`
        25的倍数都可以拆成至少两个5，但是其中的一个已经包含在n/5中了，所以+n/25是加上另一个5。
        25的倍数如果可以拆成三个5，那么一定是125的倍数，前两个5已经被n/5,n/25包含了，最后一个5包含在n/125中。以此类推。
*/
class Solution {
public:
    int trailingZeroes(int n) {
        int res = 0;

        while(n > 0){
            res += n/5;
            n /= 5;
        }

        return res;
    }
};
// 时间复杂度：**O(logn)**  
// 空间复杂度：**O(1)**
```

### [202.快乐数](../Coding\数学\202.快乐数.md)

```C++
/*
    会出现两种情况：无限循环 或者 收敛到1
    收敛到1则是快乐数
*/
class Solution {
public:
    int get(int n) { // 计算每位的平方和
        int res=0;
        while(n>0) {
            int d = n%10;
            n /= 10;
            res += d*d;
        }
        return res;
    }
    bool isHappy(int n) {
        int slow = n;
        int fast = get(n); // 初始必须是get(n)

        while(fast != 1 && slow != fast){ // 要么收敛到1，要么出现环
            slow = get(slow);
            fast = get(get(fast));
        }

        return fast == 1; // 收敛到1
    }
};
// 时间复杂度：**O(logn)**  
// 空间复杂度：**O(1)**
```

### [204.计数质数](../Coding\数学\204.计数质数.md)

```C++
/*
    求小于 n 的质数数量
    埃氏筛
    本质就是判断完x是质数，则x的倍数一定是合数，但是如果从2x、3x…开始标记会冗余，要从xx开始标记，因为在判断x之前，2x、3x…一定被标记过，例如2的倍数，3的倍数 ... 。
*/
class Solution {
public:
    int countPrimes(int n) {
        vector<int> dp(n, 1); // dp[i] 为 1 表示 i 是质数，为 0 表示 i 是合数
        int cnt = 0;

        for(int i=2; i<n; i++) {
            if(dp[i] == 1){
                cnt++;
                for(long long j=(long long)i*i; j<n; j+=i) // 注意这里一定要 long long
                    dp[j] = 0;
            }
        }

        return cnt;
    }
};
// 时间复杂度：**O(nlogn)**  
// 空间复杂度：**O(n)**
```

### [231.2的幂](../Coding\数学\231.2的幂.md)

```C++
/*
    判断 n 是否是 2 的幂次方
*/
class Solution {
public:
    bool isPowerOfTwo(int n) {

        return n>0 && (n&(n-1))==0; // 这里一定要加括号
    }
};
// 时间复杂度：**O(1)**  
// 空间复杂度：**O(1)**
```

### [233.数字-1-的个数](../Coding\数学\233.数字-1-的个数.md)

### [279.-完全平方数](../Coding\数学\279.-完全平方数.md)

```C++
/*
    求和为 n 的完全平方数的最少数量
    动态规划
    dp[i] 表示和为 i 的完全平方数的最少数量
*/
class Solution {
public:
    int numSquares(int n) {
        vector<int> dp(n+1,0); // dp[i]表示和为i的完全平方数的最少数量

        for(int i=1;i<=n;i++) {
            dp[i] = i;          // 最坏的情况，拆成多个 1
            for(int j=1;j*j<=i;j++)
                dp[i] = min(dp[i],dp[i-j*j]+1); // 所有的情况可以分为第一个数选1，选2，选3，...，选k（最大可选），总的k种情况，k种情况中个数最少的就是答案。
        }

        return dp[n];
    }
};
// 时间复杂度：**O(n*sqrt(n))**  
// 空间复杂度：**O(n)**
```

### [645. 错误的集合](../Coding\数学\645.-错误的集合.md)

```C++
/*
    1、排序：排序后，相等的两个数字将会连续出现。此外，检查相邻的两个数字是否只相差1可以找到缺失的数字

    2、哈希表：哈希每个该出现的数字，出现了两次的就是重复的，出现了零次的就是缺失的

    3、分组异或: 一个数和它本身进行异或运算结果为 `0`，`0`和任何数的异或结果为任何数
*/
//                              分组异或
class Solution {
public:
    vector<int> findErrorNums(vector<int>& nums) {
        int xor1=0,xor2=0,xor0=0;
        for(int i:nums)
            xor0 ^= i;
        for(int i=1;i<nums.size()+1;i++)
            xor0 ^= i;  
        int wei = xor0 & ~(xor0-1); // 注意这种获取最后一位1的方式。xor0-1一定会把最后一位1变为0，然后~(xor0-1)就是只有最后一位为1，其余位变为原来的反，最后与上xor0的话，就得到只有最后一位为1，其余位为0
        for(int i:nums)
            if((i&wei) == 0)// 注意这里(i&wei)一定要有括号
                xor1 ^= i;
            else
                xor2 ^= i;
        for(int i=1;i<nums.size()+1;i++)
            if((i&wei) == 0)
                xor1 ^= i;
            else
                xor2 ^= i;
        for(int i:nums)
            if(i==xor2)
                return {xor2,xor1};
        return {xor1,xor2};
    }
};
// 时间复杂度：**O(n)**  
// 空间复杂度：**O(1)**
```

### [1037.-有效的回旋镖](../Coding\数学\1037.-有效的回旋镖.md)

```C++
/*
    判断斜率，斜率不相等就不共线，就能组成回旋镖  
    `第1个点和第2个点分别和第0个点求斜率，如果这两个斜率不相等，就是不共线，就能组成回旋镖`
*/
class Solution {
public:
    bool isBoomerang(vector<vector<int>>& points) {
        int dx1 = points[1][0] - points[0][0];
        int dy1 = points[1][1] - points[0][1];
        int dx2 = points[2][0] - points[1][0];
        int dy2 = points[2][1] - points[1][1];
        return dy2*dx1 != dx2*dy1;  // 等价于dy1/dx1 == dy2/dx2 , 使用交叉相乘不用考虑分母为0
    }
};
// 时间复杂度：**O(1)**  
// 空间复杂度：**O(1)**
```

### [剑指Offer-17.打印从1到最大的n位数](../Coding\数学\剑指Offer-17.打印从1到最大的n位数.md)

```C++
/*
    全排列
*/
class Solution {
public:
    void dfs(vector<string>& res,string& tmp,int idx) {// 注意需要按引用传递
        if(idx == tmp.size()) { // 考虑完tmp的所有位
            int start = 0;
            for(int i=0;i<tmp.size();i++)// 去除高位无效0
                if(tmp[i]=='0')
                    start++;
                else
                    break;
            res.push_back(tmp.substr(start));
            return;
        }
        for(int i=0;i<=9;i++){ // tmp的第idx位有10种选择
            tmp[idx] = (char)(i+'0'); // 注意这种字符构造方法
            dfs(res,tmp,idx+1);
        }
    }
    vector<string> printNumbers(int n) {
        vector<string> res;
        string tmp(n,'0');// 注意这种构造方法
        dfs(res,tmp,0);

        return res;
    }
};
```

### [剑指Offer-44.数字序列中某一位的数字](../Coding\数学\剑指Offer-44.数字序列中某一位的数字.md)

```C++
/*
    求数字序列中的第n位是[0-9]中的哪一个。
    首先找出第n位所在的数字，然后再在这个数字中找第n位。
*/
class Solution {
public:
    int findNthDigit(int n) {
        // 首先看n所在的数字是几位数的
        int digit = 1; // 几位数
        long start = 1; // 几位数对应的起始数
        long count = 9; // 几位数的所有数拼起来共多少位
        while(n>count) {// 第一次循环查看一位的所有数，第二次循环看两位的所有数，第三次循环看三位的所有数。。。
            n -= count;
            digit += 1;
            start *= 10;
            count = 9*digit*start;// 两位数从10到99，总的90个数，每个数是两个字符，就是180
        }

        // 然后找出n所在的数
        long num = start + (n-1)/digit;// 第n位是下标n-1，(n-1)/digit计算前n位包含多少个完整的数，加上start得到的是以前n位除以digit余下的位数开始的完整的数
        string nums = to_string(num);// 第n位所在的数字转换为字符串

        // 接着从n所在的数上找出第n位
        int res = nums[(n-1)%digit] - '0';// 在字符串中找出第n位，因为第n位是下标n-1

        return res;
    }
};
```

### [剑指Offer-49.丑数](../Coding\数学\剑指Offer-49.丑数.md)

```C++
/*
    丑数是只包含质因数 2、3 和/或 5 的正整数；1 是丑数。
*/
// 判断一个数是否是丑数
class Solution {
public:
    bool isUgly(int num) {
        if(num<1)       // 注意这个条件
            return false;
        while(num%2==0)
            num/=2;
        while(num%3==0)
            num/=3;
        while(num%5==0)
            num/=5;

        return num==1;
    }
};
// 求从小到大的第n个丑数
class Solution {
public:
    int nthUglyNumber(int n) {
        vector<int> dp(n);// dp[i]表示第i个丑数
        dp[0] = 1;

        int a=0,b=0,c=0;
        for(int i=1;i<n;i++) {// 计算n个丑数
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
// 时间复杂度：**O(logn)**  
// 空间复杂度：**O(1)**
```

### [剑指Offer-60.n个骰子的点数](../Coding\数学\剑指Offer-60.n个骰子的点数.md)

```C++
/*
    掷出 n 个色子，返回所有点数总和的概率
*/
class Solution {
public:
    vector<double> dicesProbability(int n) {
        vector<int> dp(67,0);// dp[i]表示骰子 点数和为i 的次数

        for(int i=1;i<=6;i++)// 第一个骰子
            dp[i]++;
        for(int i=2;i<=n;i++)// 枚举每个骰子
            for(int j=i*6;j>=i;j--){// 倒序枚举每种状态
                dp[j] = 0;   // 因为当前骰子不可能掷出0，所以上次的dp[j]不可能用来转移得到这次的dp[j]，所以直接置0，
                for(int cur=1;cur<=6;cur++)// 当前次骰子点数
                    if(j-cur >= i-1)  // 大于等于i-1的含义是掷当前骰子之前的最小点数是i-1，也就是之前的所有骰子都为1
                        dp[j] += dp[j-cur];// 完全背包，状态转移
            }

        vector<double> res;
        int all = pow(6,n);// 总的情况数
        for(int i=n;i<=6*n;i++)
            res.push_back(dp[i]*1.0/all);

        return res;
    }
};
```

### [剑指Offer-62.圆圈中最后剩下的数字](../Coding\数学\剑指Offer-62.圆圈中最后剩下的数字.md)

```C++
/*
    从 0 号成员起开始计数，排在第 target 位的成员离开圆桌，且成员离开后从下一个成员开始计数
*/
class Solution {
public:
    int iceBreakingGame(int num, int target) {
        int f = 0; // 只有1个人时最后剩下的是第0个

        for (int i = 2; i <= num; ++i) { // 递推到有num个人是最后剩下的是第几个
            f = (target + f) % i; // 每次target位离开，然后重新开始计数，所以要 加target 然后 除余 当前的人数
        }

        return f;
    }
};
```

### [面试题-16.07.-最大数值](../Coding\数学\面试题-16.07.-最大数值.md)

```C++
/*
    找出两个数字a和b中最大的那一个
    最大的是：(|a-b| + a + b) / 2 
*/
class Solution {
public:
    int maximum(int a, int b) {
        long c = a;
        long d = b;
        int res = (int) ( (fabs(c-d) + c + d)/2 );  // 注意这里要先除以2再转换为int
        return res;
    }
};
```

## 11、数组 {#customname11}

### easy[53.最大子序和](../Coding\数组\简单\53.最大子序和.md)

```C++
/*
    求 和最大的 连续子数组
    动态规划
    f(i−1) = pre表示以上一个数结尾的连续子数组的最大和
    状态转移：f(i)=max{f(i−1)+nums[i],nums[i]}
        每个元素可以自成子数组  或者  和前面的元素构成子数组
*/
class Solution {
public:
    int maxSubArray(vector<int>& nums) {
        int pre=0; // f(i−1) = pre表示以上一个数结尾的连续子数组的最大和
        int maxAns = nums[0]; // maxAns表示要贪心得到的全局最大

        for(const auto& x:nums){
            pre = max(pre+x,x); // f(i)=max{f(i−1)+nums[i],nums[i]}
            maxAns = max(maxAns,pre); // 贪心
        }

        return maxAns;
    }
};
// 时间复杂度：**O(n)**  
// 空间复杂度：**O(1)**
```

### easy[88.合并两个有序数组](../Coding\数组\简单\88.合并两个有序数组.md)

```C++
/*
    合并两个非递减数组，合并后也为非递减
    使用双指针反向遍历，将`nums1`有效位中的当前元素和`nums2`的当前元素中较大的复制到`nums1`的最后
*/
class Solution {
public:
    void merge(vector<int>& nums1, int m, vector<int>& nums2, int n) {
        int tail1 = m-1;
        int tail2 = n-1;
        int tail = m+n-1;

        while(tail != tail1)      //注意终止条件，追上了说明nums2已经合并进来了
        {
            if(tail1 >= 0 && nums1[tail1] > nums2[tail2])
                nums1[tail--] = nums1[tail1--];
            else
                nums1[tail--] = nums2[tail2--];
            // tail1<0,说明nums1已经先遍历完了，如果nums2先遍历完，肯定就追上了
        }
    }
};
// 时间复杂度：**O(n+m)**  
// 空间复杂度：**O(1)**
```

### easy[122.买卖股票的最佳时机-II](../Coding\数组\简单\122.买卖股票的最佳时机-II.md)

```C++
/*
    动态规划
        `dp0[i]` 表示第 `i` 天交易完后  手里`没有`股票     的最大利润
        `dp1[i]` 表示第 `i` 天交易完后  手里持`有一支股票`  的最大利润 
*/
class Solution {
public:
    int maxProfit(vector<int>& prices) {
        int n = prices.size();
        int dp0 = 0;            // dp0 表示今天交易完后 手里没有股票      的最大利润，
        int dp1 = -prices[0];   // dp1 表示今天交易完后 手里持有一支股票  的最大利润

        for (int i = 1; i < n; ++i) {
            int newdp0 = max(dp0, dp1 + prices[i]); // 今天没有 可能是 昨天也没有 或者 昨天有但今天卖了
            int newdp1 = max(dp1, dp0 - prices[i]); // 今天有   可能是 昨天就有   或者 昨天没有但今天买了

            dp0 = newdp0;
            dp1 = newdp1;
        }

        return dp0;
    }
};
// 时间复杂度：**O(n)**  
// 空间复杂度：**O(1)**
```

### medium[309.最佳买卖股票时机含冷冻期](../Coding\数组\中等\309.最佳买卖股票时机含冷冻期.md)

```C++
/*
    动态规划
    `卖出股票后，你无法在第二天买入股票` (即冷冻期为 1 天)
*/
class Solution {
public:
    int maxProfit(vector<int>& prices) {
        if(prices.empty())
            return 0;
        int dp[3];
        dp[0] = -prices[0]; // 今天结束时  手上  持有股票                             的最大收益
        dp[1] = 0;          // 今天结束时  手上不持有股票，并且后一天``处于``冷冻期     的累计最大收益
        dp[2] = 0;          // 今天结束时  手上不持有股票，并且后一天``不处于``冷冻期   的累计最大收益

        for(int i=1;i<prices.size();i++) {
            int dp0 = dp[0];
            int dp1 = dp[1];
            int dp2 = dp[2];
            dp[0] = max(dp0,dp2-prices[i]); // 今天有 可能是 昨天就有 或者 昨天没有且今天不处于冷冻期然后买了
            dp[1] = dp0 + prices[i];        // 今天没有且后一天处于冷冻期  只可能是  昨天有，今天卖了
            dp[2] = max(dp1,dp2);           // 今天没有且后一天不处于冷冻期  只可能是 今天没卖，也就是昨天就没有  
        }

        return max(dp[1],dp[2]);
    }
};
// 时间复杂度：**O(n)**  
// 空间复杂度：**O(1)**
```

### easy[169.多数元素](../Coding\数组\简单\169.多数元素.md)

```C++
/*
    求一个数组的众数
    直接计数排序时间和空间复杂度都为n，投票法空间复杂度能降为1
    Boyer-Moore投票法：
        - 维护一个候选众数`candidate`及其当前的得票数`count`，遍历每个元素为候选众数投票
        - 如果当前元素`等于`候选，`count+1`，`不等`则`-1`，`count`减到`0`则`更换候选众数`

        使用条件：众数出现的次数大于`floor(n/2)`，因为算法核心是其他人都投反对票，自己人的数量得多余其他人
*/
class Solution {
public:
    int majorityElement(vector<int>& nums) {
        int candidate = -1, count = 0; // 初始化候选众数为-1，得票数为0

        for(int i:nums){
            if(i != candidate){ // 当前元素不等于候选众数
                count--;

                if(count<0){ // 更换众数
                    candidate = i;
                    count = 1;
                }
            }
            else              // 当前元素等于候选众数
                count++;
        }

        return candidate;
    }
};
// 时间复杂度：**O(n)**  
// 空间复杂度：**O(1)**
```

### easy[448.找到所有数组中消失的数字](../Coding\数组\简单\448.找到所有数组中消失的数字.md)

```C++
/*
    原地修改：
    值域`[1,n]`，下标`[0,n-1]`，
    如果区间`[1,n]`内每个数都在`nums`中，那么所有的值都减`1`后和所有的下标能完美匹配上，
    哪个下标`i`没有被匹配，说明数`i+1`不存在于`nums`中，可以通过对`nums[i]`乘以`-1`来表示下标`i`已经被匹配
*/
class Solution {
public:
    vector<int> findDisappearedNumbers(vector<int>& nums) {
        for(int i:nums)
            if(nums[abs(i)-1]>0)
                nums[abs(i)-1] *= -1;// 这里使用abs的原因是当前的i可能已经被乘-1了

        vector<int> result;
        for(int i = 0;i<nums.size();++i) {
            if(nums[i]>0)
                result.push_back(i+1);
        }

        return result;
    }
};
// 时间复杂度：**O(n)**  
// 空间复杂度：**O(1)**
```

### easy[581.最短无序连续子数组](../Coding\数组\简单\581.最短无序连续子数组.md)

```C++
/*
    无序子数组中 最小的元素的正确位置可以决定左边界，
                最大的元素的正确位置可以决定右边界。
*/
class Solution {
public:
    int findUnsortedSubarray(vector<int>& nums) {
        // 正向遍历 找最小元素
        int min_ = numeric_limits<int>::max();
        int flag = 0;
        for (int i = 0; i < nums.size()-1 ; i++){
            if (nums[i] > nums[i+1])
                flag = 1; // 发现非升序序列，开始记录最小值
            if (flag == 1)
                min_ = min(min_, nums[i+1]);
        }
        // 正向遍历 找最小元素的正确位置
        int l;
        for(l=0;l<nums.size();l++){
            if(nums[l]>min_)
                break; // nums[l]大于min_，说明min_的正确位置应该是l
        }

        // 反向遍历 找最大元素
        int max_ = numeric_limits<int>::min();
        flag = 0;
        for(int i=nums.size()-1;i>0;i--){
            if(nums[i]<nums[i-1])
                flag=1; // 发现非降序序列，开始记录最大值
            if(flag==1)
                max_ = max(max_,nums[i-1]);
        }
        // 反向遍历 找最大元素的正确位置
        int r;
        for(r=nums.size()-1;r>=0;r--){
            if(nums[r]<max_)
                break; // nums[r]小于max_，说明max_的正确位置应该是r
        }
        
        return r-l>0 ? r-l+1 : 0; // 如果不存在无序子序列，则l=nums.size()-1,r=0,相减就是负数。
    }
};
// 时间复杂度：**O(n)**  
// 空间复杂度：**O(1)**
```

### easy[665.非递减数列](../Coding\数组\简单\665.非递减数列.md)

```C++
/*
    判断在`最多`改变`1`个元素的情况下，能否变为`非递减`数组
    `nums[i] > nums[i+1]`时出现了递减，此时有两种情况：  
    `nums[i-1] <= nums[i+1]`，此时只需将`nums[i]`调整为`nums[i+1]`，也就是`i`降下来。  
    `nums[i-1] > nums[i+1]`，此时需要把将`nums[i+1]`调整为`nums[i]`，也就是`i+1`提上去。
*/
class Solution {
public:
    bool checkPossibility(vector<int>& nums) {
        if (nums.size() <= 2)
            return true;

        int count = 0; // 总的调整次数
        for (int i = 0; i < nums.size()-1; i++) {
            if (nums[i] > nums[i+1]) { // 出现递减，需要调整
                count++;

                if (i == 0)                // 注意这里
                    nums[i] = nums[i+1];
                else if (nums[i-1] > nums[i+1])
                    nums[i+1] = nums[i];
                else if (nums[i-1] <= nums[i+1])
                    nums[i] = nums[i+1];
            }
        }

        return count <= 1;
    }
};
// 时间复杂度：**O(n)**  
// 空间复杂度：**O(1)**
```

### easy[674.最长连续递增序列](../Coding\数组\简单\674.最长连续递增序列.md)

```C++
/*
    求最长连续递增序列的长度
    动态规划
    维护一个变量`temp`记录连续递增的次数，遇到递减就置`1`，时刻维护`temp`所能达到的最大值`ans`
*/
class Solution {
public:
    int findLengthOfLCIS(vector<int>& nums) {
        if (nums.size() <= 1)
            return nums.size();

        int ans = 1, temp = 1;
        for (int i = 1; i < nums.size(); i++) {
            if (nums[i] > nums[i-1])
                temp++;     //   递增就 temp++
            else
                temp = 1;   // 非递增就 temp置1
            ans = max(ans,temp);
        }

        return ans;
    }
};
// 时间复杂度：**O(n)**  
// 空间复杂度：**O(1)**
```

### easy[724.寻找数组的中心索引](../Coding\数组\简单\724.寻找数组的中心索引.md)

```C++
/*
    定义变量`sum`表示 数组总和，  
    定义`suml`表示 当前下标左侧所有元素的和，  
    对于每个下标，判断 左边元素和 是否等于 右边元素和
*/
class Solution {
public:
    int pivotIndex(vector<int>& nums) {
        int sum = 0, suml = 0;

        for (int i:nums)
            sum += i; // 求总和

        for (int i = 0; i < nums.size(); i++) {
            if (suml == sum - suml - nums[i])
                return i; // 左边或者右边没有元素则默认和为0
            else
                suml += nums[i];
        }

        return -1;
    }
};
// 时间复杂度：**O(n)**  
// 空间复杂度：**O(1)**
```

### easy[914.卡牌分组](../Coding\数组\简单\914.卡牌分组.md)

```C++
/*
    一个数组由多种整数组成，每种整数可能出现多次，求这些整数出现次数的最大公约数
    遍历`频数数组`，依次用`前i-1`个元素的最大公约数和当前的`第i个`元素求最大公约数，最终结果为整个频数数组的最大公约数
*/
class Solution {
public:
    int cnt[10000];
    bool hasGroupsSizeX(vector<int>& deck) {
        for (int i:deck)
            cnt[i]++; // 存储频数

        int g = -1; // 最大公约数
        for (int i = 0; i < 10000; i++)
            if (cnt[i] != 0) {  // 重要注意
                if (g == -1)
                    g = cnt[i]; // 之前没有元素，当前元素设为最大公约数
                else
                    g = gcd(g, cnt[i]); // 依次计算当前元素cnt[i]和之前所有元素的最大公约数g之间的最大公约数
            }

        return g >= 2; // 是否存在除1外的最大公约数
    }
    int gcd(int a,int b) { // 辗转相除法：以除数和余数反复做除法运算，当余数为 0 时，取当前算式除数为最大公约数
        if (b == 0)
            return a;
        else
            return gcd(b, a%b);
    }
};
// 时间复杂度：**O(nlogc)**  
// 空间复杂度：**O(n+c)**
```

### easy[剑指-Offer-40.最小的k个数](../Coding\数组\简单\剑指-Offer-40.最小的k个数.md)

```C++
/*
    快速排序
*/
// 基于快速排序的选择算法
class Solution {
public:
    void quicksort(vector<int>& nums,int L,int R,int k){
        if(L>=R) // 递归终止条件
            return;
        int P = rand()%(R-L+1)+L; // 随机选择主元
        swap(nums[P],nums[R]); // 主元交换到最右边
        int i = L-1;
        for(int j=L;j<=R-1;j++) // 从左开始遍历
            if(nums[j]<=nums[R])
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
        // srand((unsigned)time(nullptr)); // 设置随机数种子
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
        // srand((unsigned)time(nullptr));
        Qsort(nums,0,nums.size()-1);
        return nums[nums.size()-k];
    }
    void Qsort(vector<int>& nums,int L, int R){
        if(L>=R)
            return;
        int P = rand() % (R-L+1) + L;
        swap(nums[P],nums[R]);
        int i = L-1;
        for(int j=L;j<=R-1;j++)
            if(nums[j]<=nums[R])
                swap(nums[++i],nums[j]);
        swap(nums[++i],nums[R]);
        Qsort(nums,L,i-1);
        Qsort(nums,i+1,R);
    }
};
// 时间复杂度：**O(nlogn)**  
// 空间复杂度：**O(logn)**
```

### easy[剑指-Offer-53-I.在排序数组中查找数字](../Coding\数组\简单\剑指-Offer-53-I.在排序数组中查找数字.md)

```C++
/*
    在一个非递减的数组`nums`中，找出目标值`target`出现的`开始位置`和`结束位置`。  
    二分查找：执行两次二分查找，分别查找`第一个大于等于target`的下标、`第一个大于target`的下标。
*/
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
int binary_search_nonrecursion(vector<int> List,int n) {
    int L = 0, R = List.size()-1;
    while(L <= R)
    {
        int mid = (L + R)/2;
        if(List[mid] < n)
            L = mid + 1;
        else if(List[mid] > n)
            R = mid - 1;
        else 
            return mid;
    }
    return -1;
}
// 时间复杂度：**O(logn)**  
// 空间复杂度：**O(1)**
```

### easy[剑指-Offer-53-II.0～n-1中缺失的数字](../Coding\数组\简单\剑指-Offer-53-II.0～n-1中缺失的数字.md)

```C++
/*
    数组`nums`长度为`n`，按升序存放了`[0,n]`内的`n`个数，找出`[0,n-1]`中没有出现的数
*/
class Solution {
public:
    int missingNumber(vector<int>& nums) {
        if(nums.empty())
            return 0;

        int l = 0, r = nums.size() - 1;
        while(l <= r){
            int mid = (l + r) / 2;
            if(nums[mid] != mid) 
                r = mid - 1; // 只要下标 i 不等于值 nums[i] 就到左边去继续找，最终找到的就是第一个和值不相等的下标。
            else 
                l = mid + 1;
        }

        return l;
    }
};
// 时间复杂度：**O(logn)**  
// 空间复杂度：**O(1)**
```

### medium[31.下一个排列](../Coding\数组\中等\31.下一个排列.md)

```C++
/*
    返回数组在`字典序`中的`下一个排列`。
        - 首先从后往前找到第一个递减的元素`nums[i]`，  
        - 然后再一次从后往前找到第一个比`nums[i]`大的元素`nums[j]`，  
        - 交换`nums[i]`和`nums[j]`，  
        - 接着把下标`i`后面的所有元素`reverse`。  
*/
class Solution {
public:
    void nextPermutation(vector<int>& nums) {
        // 首先从后往前找到第一个递减的元素`nums[i]`
        int i = nums.size()-2;
        while(i>=0 && nums[i]>=nums[i+1])
            i--;
        // 然后再一次从后往前找到第一个比`nums[i]`大的元素`nums[j]`
        if(i>=0) {
            int j = nums.size()-1;
            while(nums[i]>=nums[j])
                j--;
            swap(nums[i],nums[j]); // 交换`nums[i]`和`nums[j]`
        }

        reverse(nums.begin()+i+1,nums.end()); // 接着把下标`i`后面的所有元素`reverse`
    }
};
// 时间复杂度：**O(n)**  
// 空间复杂度：**O(1)**
```

### medium[39.组合总和](../Coding\数组\中等\39.组合总和.md)

```C++
/*
    使数字和为目标数`target`的 所有 `不同组合`，每个元素可以无限次使用
    搜索回溯
*/
class Solution {
public:
    void dfs(vector<int>& candidates, int target, vector<vector<int>>& ans, vector<int>& tmp, int idx){
        // candidates:      可选元素数组
        // target:          还需要组合的目标值
        // ans:             存储最终结果
        // tmp:         保存当前选择过的数字
        // idx:             当前处理的数字在 candidates 中的下标
        if(idx == candidates.size())// 如果当前处理数字下标为n，表示已经用了所有可选的数字
            return;
        if(target==0){// 如果还差0得到target，就说明找到了一个组合，其和为target，将这个组合加入结果
            ans.emplace_back(tmp);
            return;
        }

        // 不使用下标为Idx的元素
        dfs(candidates, target, ans, tmp, idx+1);
        // 使用下标为Idx的元素
        if(candidates[idx] <= target) {
            tmp.emplace_back(candidates[idx]);      // 下标为Idx的数字加入tmp表示使用
            dfs(candidates, target-candidates[idx], ans, tmp, idx); // 使用Idx后，目标target减小，但是数字可以重复使用，因此递归还是使用Idx
            tmp.pop_back();
        }
    }
    vector<vector<int>> combinationSum(vector<int>& candidates, int target) {
        vector<vector<int>> ans;
        vector<int> tmp;
        dfs(candidates,target,ans,tmp,0);
        return ans;
    }
};
// 时间复杂度：**O(S)** S为所有可行解的长度之和  
// 空间复杂度：**O(target)** 除答案数组外，空间复杂度取决于递归的栈深度，在最差情况下需要递归 O(target) 层
```

### medium[46.全排列](../Coding\数组\中等\46.全排列.md)

```C++
/*
    对于一个`不含重复数字`的数组`nums`，返回其`所有可能的全排列`。
    回溯
*/
class Solution {
public:
    void dfs(vector<int>& nums,vector<vector<int>>& res,int idx){
        /*
            nums：      存储中间排列的数组，这里直接在原始数组中操作
            res：       存储最终的结果
            idx：       当前所考虑的元素在 nums 的下标
        */
        if(idx==nums.size()) { // 如果已经挑选过所有元素，就得到了一个完整的排列，因为没有约束条件，每个完整的排列都符合要求，直接添加到结果
            res.push_back(nums);
            return;
        }

        for(int i = idx;i<nums.size();i++) {
            swap(nums[i],nums[idx]);
            dfs(nums,res,idx+1);
            swap(nums[i],nums[idx]);
        } // 实际上第一层递归中的for循环了n次，分别对应n个元素每个元素都放到下标为0的位置一次。
    }
    vector<vector<int>> permute(vector<int>& nums) {
        vector<vector<int>> res;
        dfs(nums,res,0);
        return res;
    }
};
// 时间复杂度：**O(n x n!)**  
// 空间复杂度：**O(n)**
```

### medium[48.旋转图像](../Coding\数组\中等\48.旋转图像.md)

```C++
/*
    将图像`顺时针旋转90度`
    使用翻转代替旋转  
        顺时针旋转90° = 上下翻转+主对角线翻转 = 主对角线翻转+左右翻转  
*/
class Solution {
public:
    void rotate(vector<vector<int>>& matrix) {
        int n = matrix.size();
        // 上下翻转
        for(int i=0;i<n/2;++i) // 注意i
            for(int j=0;j<n;++j)
                // i <---> n-i-1  
                swap(matrix[i][j],matrix[n-1-i][j]);
        // 主对角线翻转
        for(int i=0;i<n;++i)
            for(int j=0;j<i;++j) // 注意j<i
                swap(matrix[i][j],matrix[j][i]);
    }
};
// 时间复杂度：**O(n²)**  
// 空间复杂度：**O(1)**
```

### medium[189.旋转数组](../Coding\数组\中等\189.旋转数组.md)

```C++
/*
    将数组中的元素向右轮转 k 个位置
        1、整体    原地翻转  
        2、前k个   原地翻转  
        3、后n-k个 原地翻转  
*/
class Solution {
public:
    void rotate(vector<int>& nums, int k) {
        k %= nums.size();
        // 整体 原地翻转
        reverse(nums.begin(), nums.end());
        // 前k个 原地翻转
        reverse(nums.begin(), nums.begin()+k);
        // 后n-k个 原地翻转
        reverse(nums.begin()+k, nums.end());
    }
};
// 时间复杂度：**O(n)**  
// 空间复杂度：**O(1)**  
```

### medium[55.跳跃游戏](../Coding\数组\中等\55.跳跃游戏.md)

```C++
/*
    判断是否能够到达最后一个下标
    贪心
*/
class Solution {
public:
    bool canJump(vector<int>& nums) {
        int maxdistance = nums[0]; // 当前遍历过的位置能跳到的最远下标

        for(int i = 0; i < nums.size(); i++) {
            if(i <= maxdistance) { // 遍历每一个元素，如果当前位置是之前能跳到的，就考虑当前位置
                maxdistance = max(maxdistance, i + nums[i]); // 用当前位置能跳的最远更新历史最远

                if(maxdistance >= nums.size() - 1) // 如果历史最远超过了n-1，表示一定能跳到n-1
                    return true;
            }
            else // 如果当前位置是之前的每个位置都到达不了的，能执行到这，必然没有执行上面if中的return true，当前都到不了，更到不了n-1
                return false;
        }

        return false;
    }
};
// 时间复杂度：**O(n)**  
// 空间复杂度：**O(1)**
```

### medium[56.合并区间](../Coding\数组\中等\56.合并区间.md)

```C++
/*
    合并所有重叠的区间
*/
class Solution {
public:
    void quicksort(vector<vector<int>>& nums,int L,int R){
        if(L>=R)
            return ;
        int P = rand()%(R-L+1) + L;
        swap(nums[P],nums[R]);
        int i = L-1;
        for(int j=L;j<=R-1;j++)
            if(nums[j][0]<nums[R][0] || (nums[j][0]==nums[R][0] && nums[j][1]<=nums[R][1]))
            // if(nums[j]<=nums[R])
                swap(nums[++i],nums[j]);
        swap(nums[++i],nums[R]);
        quicksort(nums,L,i-1);
        quicksort(nums,i+1,R);
    }
    vector<vector<int>> merge(vector<vector<int>>& intervals) {
        // srand((unsigned)time(nullptr));
        quicksort(intervals,0,intervals.size()-1);

        vector<vector<int>> res{intervals[0]};
        for(int i=1;i<intervals.size();i++){
            int L = intervals[i][0];
            int R = intervals[i][1];

            if(L<=res.back()[1])  // 合并
                res.back()[1] = max(R,res.back()[1]);
            else                  // 添加
                res.push_back(intervals[i]);
        }

        return res;
    }
};
// 时间复杂度：**O(nlogn)**  
// 空间复杂度：**O(logn)**
```

### medium[75.颜色分类](../Coding\数组\中等\75.颜色分类.md)

```C++
/*
    整数`0、 1 和 2`分别表示`红色、白色和蓝色`，对数组排序
    双指针
*/
class Solution {
public:
    void sortColors(vector<int>& nums) {
        int i = 0; // i 的左边全放 0
        int k = nums.size()-1; // k 的右边全放 2

        for(int j = i; j <= k; j++) { // 一次遍历，将0换到开头，2换到结尾，注意这里终止条件为 i<=k
            while(j<=k && nums[j]==2) // 如果换到j的也是一个2，此时如果j跳去处理下一个，这个2就留在了前面，会出错，因此需要一直换，直到换回来的不是2
                swap(nums[k--],nums[j]);

            // 此时nums[j]不是0就是1
            if(nums[j] == 0)
                swap(nums[i++],nums[j]);
        }
    }
};
// 时间复杂度：**O(n)**  
// 空间复杂度：**O(1)**
```

### medium[78.子集](../Coding\数组\中等\78.子集.md)

```C++
/*
    对于元素`互不相同`的一个整数数组`nums`，返回其所有可能的`子集`。
    回溯
*/
class Solution {
public:
    void dfs(vector<vector<int>>& res,vector<int>& tmp,int idx,vector<int>& nums){
        /*
            res:    存储结果
            tmp:    存储当前枚举的集合
            idx:    当前枚举所考虑的 nums 元素下标
            nums:   需要枚举子集的原始集合
        */
        if(idx == nums.size()){ // 如果已经考虑了所有的元素，就将得到的子集tmp加入结果，然后递归返回
            res.push_back(tmp);
            return;
        }

        // 子集中不包含当前元素
        dfs(res,tmp,idx+1,nums); // 在不包含当前元素 的情况下考虑后面的元素
        // 子集中包含当前元素
        tmp.push_back(nums[idx]);
        dfs(res,tmp,idx+1,nums); // 在包含当前元素   的情况下考虑后面的元素
        tmp.pop_back(); // 返回上一层递归之前，清除之前的操作
    }
    vector<vector<int>> subsets(vector<int>& nums) {
        vector<vector<int>> res;
        vector<int> tmp;
        dfs(res,tmp,0,nums);
        return res;
    }
};
// 时间复杂度：**O(nx2ⁿ)**  
// 空间复杂度：**O(n)**
```

### medium[79.单词搜索](../Coding\数组\中等\79.单词搜索.md)

```C++
/*
    判断`word`是否存在于网格中
    深度优先 + 回溯剪枝
*/
class Solution {
public:
    vector<vector<int>> direction{{1,0},{0,1},{-1,0},{0,-1}};   // 访问方向
    bool check(vector<vector<char>>& board,vector<vector<int>>& visited,int i,int j,string& word,int k){
        /*
            board：     输入的字符数组
            visited：   用于标识元素是否已经被访问过
            i、j：      开始递归查找word的起始字符在输入数组中的行列下标   
            word：      需要查找的字符串
            k：         当前递归需要查找的字符在word中的下标
        */
        if(board[i][j]!=word[k]) // 如果当前字符不是要找的下一个字符word[k]
            return false;
        else if(k==word.size()-1) // 如果已经找到word的所有字符
            return true;

        bool result = false;

        // 标记当前字符已使用，然后递归
        visited[i][j] = 1;
        for(vector<int> d:direction){ // 对于当前字符的所有方向
            int newi = i+d[0];
            int newj = j+d[1];

            if(0<=newi && newi<=board.size()-1 && 0<=newj && newj<=board[0].size()-1) // 如果没有越界
                if(visited[newi][newj] != 1) {// 如果当前方向(newi,newj)上的这个字符没有被访问过
                    bool flag = check(board,visited,newi,newj,word,k+1); // 从当前方向开始顺序找word中下标为[k+1,n-1]的部分，找到返回true，否则返回false
                    if(flag){
                        result = true;
                        break; // 如果在一个方向上找到了，就不用考虑其他方向，每层递归返回后都不再处理剩下的方向，一直递归返回到首次调用处。
                    }
                }
        }
        // 标记当前字符未使用，相当于回溯里的pop操作
        visited[i][j] = 0; // (i,j)可能在以其它字符开始的搜索顺序中被再次用到，所以需要重新设为false

        // 设置result最后再返回而不在设置处直接返回是因为回溯需要撤销之前的选择，即 visited[i][j] = 0;
        return result; // 从当前字符开始找，没有顺序找到word中下标为[k+1,n-1]的所有字符则返回false，有一个方向找到了，就返回true。        
    }    
    bool exist(vector<vector<char>>& board, string word) {
        int h = board.size();
        int w = board[0].size();
        vector<vector<int>> visited(h,vector<int>(w,0)); //用于标识元素是否已经被访问过

        // 只有当board中每种字符的数量都大于word时，才去递归搜索
        unordered_map<char,int> mp;
        for(int i=0;i<h;i++)
            for(int j=0;j<w;j++)
                mp[board[i][j]]++;
        for(char i:word)
            if(--mp[i]<0)
                return false;

        // 递归搜索
        for(int i=0;i<h;i++)
            for(int j=0;j<w;j++)
                if(board[i][j] == word[0]) { // 遍历每个位置，等于word中第一个字符才去继续查找
                    bool flag = check(board,visited,i,j,word,0);
                    if(flag)
                        return true;
                }

        return false;
    }
};
// 时间复杂度：**O( MN⋅(3的L次方) )**，L 为字符串 word 的长度  
// 空间复杂度：**O( MN )**，额外开辟了 O(MN) 的 visited 数组
```

### medium[128.最长连续序列（代码还没看）](../Coding\数组\中等\128.最长连续序列（代码还没看）.md)

### medium[152.乘积最大子数组](../Coding\数组\中等\152.乘积最大子数组.md)

```C++
/*
    求连续子数组的最大乘积
    动态规划
*/
class Solution {
public:
    int maxProduct(vector<int>& nums) {
        int maxc = INT_MIN;
        int minlast = 1; // 表示 以上一个元素结尾 的乘积 最小 子数组的乘积
        int maxlast = 1; // 表示 以上一个元素结尾 的乘积 最大 子数组的乘积

        for(int i:nums) {
            int max_ = maxlast;
            int min_ = minlast; // 注意这里要先缓存，避免求minlast时使用已经更新过的maxlast
            // 状态转移
            maxlast = max(i, max(i*min_, i*max_)); // 三者取最大
            minlast = min(i, min(i*min_, i*max_)); // 三者取最小
            // 贪心
            maxc = max(maxc,maxlast);
        }

        return maxc;
    }
};
// 时间复杂度：**O(n)**  
// 空间复杂度：**O(1)**
```

### medium[198.打家劫舍](../Coding\数组\中等\198.打家劫舍.md)

```C++
/*
    不触动警报装置的情况下 ，一夜之内能够偷窃到的最高金额
    动态规划
*/
class Solution {
public:
    int rob(vector<int>& nums) {
        int steal = 0;    // 偷这家的情况下    处理完当前这家时的最大金额
        int no_steal = 0; // 不偷这家的情况下  处理完当前这家时的最大金额

        for(int i:nums) {
            int no_steal_ = no_steal;
            int steal_ = steal;

            // 状态转移
            steal = i + no_steal_; // 偷这家，上一家就不能偷
            no_steal = max(steal_, no_steal_); // 不偷这家，上一家就可偷可不偷
        }

        return max(steal,no_steal);
    }
};
// 时间复杂度：**O(n)**  
// 空间复杂度：**O(1)**
```

### medium[215.数组中的第K个最大元素](../Coding\数组\中等\215.数组中的第K个最大元素.md)

```C++
/*
    基于快速排序的选择算法
*/
class Solution {
public:
    void quicksort(vector<int>& nums,int L,int R,int k){
        if(L>=R)
            return;
        int P = rand()%(R-L+1) + L;
        swap(nums[P],nums[R]);
        int i = L-1;
        for(int j=L;j<R;j++)
            if(nums[j]<=nums[R])
                swap(nums[++i],nums[j]);
        swap(nums[++i],nums[R]);

        if(k==i) // 注意和 剑指 Offer 40. 最小的k个数.md 的区别，这里是找下标，不是找数量
            return;
        else if(k<i)
            quicksort(nums,L,i-1,k);
        else
            quicksort(nums,i+1,R,k); // 找的是下标，所以这里两处都是k
    }
    int findKthLargest(vector<int>& nums, int k) {
        // srand((unsigned)time(nullptr));
        quicksort(nums,0,nums.size()-1,nums.size() - k); // 注意从小到大排序，第k个最大元素的位置为n-k，不为k-1

        return nums[nums.size()-k];
    }
};
// 时间复杂度：**O(n)**  
// 空间复杂度：**O(logn)**
```

### medium[238.除自身以外数组的乘积](../Coding\数组\中等\238.除自身以外数组的乘积.md)

```C++
/*
    给你一个整数数组`nums`，返回数组`answer`，其中`answer[i]`等于`nums`中除`nums[i]`之外其余各元素的乘积。

    一次遍历，维护当前元素左边的累积  
    二次遍历，维护当前元素右边的累积
*/
class Solution {
public:
    vector<int> productExceptSelf(vector<int>& nums) {
        int n = nums.size();
        vector<int> res(n,1);
        int left = 1, right = 1;     // left：从左边累乘，right：从右边累乘
        
        // 一次遍历，维护当前元素左边的累积
        for(int i=0; i<n; ++i) {
            res[i] *= left;       //乘以其左边的乘积
            left *= nums[i];      //更新得到i+1左边的乘积
        }
        // 二次遍历，维护当前元素右边的累积
        for(int i=n-1; i>=0; --i) {
            res[i] *= right;      //乘以其右边的乘积
            right *= nums[i];     //更新得到i-1右边的乘积
        }

        return res;
    }
};
// 时间复杂度：**O(n)**  
// 空间复杂度：**O(1)**
```

### medium[240.搜索二维矩阵-II](../Coding\数组\中等\240.搜索二维矩阵-II.md)

```C++
/*
    在二维矩阵中查找target，矩阵：
        每行的元素`从左到右升序`排列。  
        每列的元素`从上到下升序`排列。
    从左下角或者右上角开始搜索
*/
class Solution {
public:
    bool searchMatrix(vector<vector<int>>& matrix, int target) {
        // 左下角
        int row = matrix.size()-1;
        int col = 0;

        // 从左下角开始搜索
        while(row >= 0 && col < matrix[0].size()) {
            if(matrix[row][col] > target)     // 如果当前值大于目标值，往上走
                row--;
            else if(matrix[row][col] < target)// 如果当前值小于目标值，往右走
                col++;
            else
                return true;
        }

        return false;
    }
};
// 时间复杂度：**O(m+n)**  
// 空间复杂度：**O(1)**
```

### medium[287.寻找重复数](../Coding\数组\中等\287.寻找重复数.md)

```C++
/*
    `下标有0`，但`值没有0`，也就是从下标`0`开始可以通过 `值i->nums[值i]`连接所有`值和下标不相等`的位置，相当于构成了一个`链表`，重复的元素会连接到同一个下标位置，相当于有环了，入环点就是重复的元素

    可通过快慢指针判断是否有环，并找到入环点
*/
class Solution {
public:
    int findDuplicate(vector<int>& nums) {
        int slow = 0, fast = 0; // 从下标 0 开始

        while (true) {
            slow = nums[slow];
            fast = nums[nums[fast]];
            if (slow == fast)
                break;            
        }

        slow = 0; // slow从头开始
        while (slow != fast) { // while退出时，找到入环点 
            slow = nums[slow];
            fast = nums[fast];
        }

        return slow;
    }
};
// 时间复杂度：**O(n)**  
// 空间复杂度：**O(1)**
```

### medium[322.零钱兑换](../Coding\数组\中等\322.零钱兑换.md)

```C++
/*
    求可以凑成总金额 amount 所需的`最少的硬币个数`
    动态规划:  
        `dp[i]`为组成金额`i`所需的`最少`的硬币数量
*/
class Solution {
public:
    int coinChange(vector<int>& coins, int amount) {
        vector<int> dp(amount+1, amount+100); // 状态数组中每个值设为 amount+100 这里加任意一个数都可以
        dp[0] = 0;

        for(int i=1; i<=amount; i++) { // 遍历每一个amount
            for(int j=0; j<coins.size(); j++) { // 遍历每种硬币
                if(coins[j] <= i) // 如果可以使用当前面值的硬币
                    dp[i] = min(dp[i], dp[i-coins[j]]+1); // 状态转移
            }
        }

        return dp[amount] > amount ? -1:dp[amount]; // 大于返回-1是因为硬币数最大只可能为amount，此时全选面值为1的硬币
    }
};
// 时间复杂度：**O(Sn)**  
// 空间复杂度：**O(S)**
```

### medium[347.前-K-个高频元素](../Coding\数组\中等\347.前-K-个高频元素.md)

```C++
/*
    `出现频率前k高的元素`。你可以按`任意顺序`返回
    基于快速排序的选择算法
*/
class Solution {
public:
    void quicksort(vector<pair<int,int>>& nums,int L, int R, int k){ // 注意这里一定要写引用
        if(L>=R)
            return;
        int p = rand()%(R-L+1)+L;
        swap(nums[p],nums[R]);
        int i=L-1;
        for(int j=L;j<=R-1;j++)
            if(nums[j].second<=nums[R].second) // 按频率升序排序
                swap(nums[++i],nums[j]);
        swap(nums[++i],nums[R]);

        if(k==i)
            return;
        else if(k<i)
            quicksort(nums,L,i-1,k);
        else
            quicksort(nums,i+1,R,k); // 去右边继续找下标k
    }
    vector<int> topKFrequent(vector<int>& nums, int k) {
        // 统计频率
        unordered_map<int,int> mp;
        for(int i:nums)
            mp[i]++;
        vector<pair<int,int>> arr;
        for(pair<int,int> i:mp)
            arr.push_back(i);

        // 排序
        // srand((unsigned)time(nullptr));
        quicksort(arr,0,arr.size()-1,arr.size()-k); // 频率降序的前k个就是频率升序的倒数k个

        // 输出
        vector<int> res;
        int n = arr.size();
        for(int i=n-1;i>=n-k;i--)
            res.push_back(arr[i].first);

        return res;
    }
};
// 时间复杂度：**O(n)**  
// 空间复杂度：**O(n)**
```

### medium[406.根据身高重建队列](../Coding\数组\中等\406.根据身高重建队列.md)

```C++
/*
    到这
*/
class Solution {
public:
    bool compare(const vector<int>& a,const vector<int>& b)const{
        return a[0] > b[0] || (a[0]==b[0] && a[1] < b[1]); // 按hi降序，ki升序排序
    }
    void quicksort(vector<vector<int>>& nums,int L,int R){
        if(L>=R)return;
        int P = rand()%(R-L+1)+L;
        swap(nums[R],nums[P]);
        int i = L-1;
        for(int j=L;j<R;j++)
            if(compare(nums[j],nums[R]))
                swap(nums[++i],nums[j]);
        swap(nums[++i],nums[R]);
        quicksort(nums,L,i-1);
        quicksort(nums,i+1,R);
    }
    vector<vector<int>> reconstructQueue(vector<vector<int>>& people) {
        srand((unsigned)time(nullptr));
        quicksort(people,0,people.size()-1);
        vector<vector<int>> res;
        for(vector<int> p:people)
            res.insert(res.begin()+p[1],p);//插入到i[1]的位置，前面有i[1]个人高于i，且后面再插入的人不会影响i的正确性，因为都比i矮
        return res;
    }
};
// 时间复杂度：**O(n)**  
// 空间复杂度：**O(1)**
```

### medium[416.分割等和子集](../Coding\数组\中等\416.分割等和子集.md)

```C++
/*
*/
class Solution {
public:
    bool canPartition(vector<int>& nums) {
        // 如果元素小于2，返回false
        int n = nums.size();
        if (n < 2) {
            return false;
        }
        // 计算数组的和
        int sum = 0, maxNum = 0;
        for (auto& num : nums) {
            sum += num;
            maxNum = max(maxNum, num);
        }
        // 如果和为奇数，则不可能分为等和的两部分，返回false
        if(sum%2!=0)
            return false;
        // 如果和的一半小于最大的元素，也就是说包含最大值或者不包含最大值都不肯定等于和的一半，返回false
        int target = sum / 2;
        if (maxNum > target) {
            return false;
        }

        vector<int> dp(target + 1, 0);// 初始时，dp 中的全部元素都是 false。
        dp[0] = true; // dp[i] 表示是否可以在nums中找到一个子集，和为i，可以则为true。
        for (int i = 0; i < n; i++) {// 遍历每个元素，依次考虑
            int num = nums[i];
            for (int j = target; j >= num; --j) {//需要从大到小计算，因为如果我们从小到大更新 dp 值，那么在计算 dp[j] 值的时候，dp[j−nums[i]] 已经是被更新过的状态，不再是上一次的 dp 值。
                dp[j] |= dp[j - num];
            }
        }
        return dp[target];
    }
};
// 时间复杂度：**O(n*target)**  
// 空间复杂度：**O(target)**
```

### medium[494.目标和](../Coding\数组\中等\494.目标和.md)

```C++
/*
*/
class Solution {
public:
    int findTargetSumWays(vector<int>& nums, int S) {
        vector<int> dp(2001,0); // dp[i] 表示当前考虑范畴内的所有元素可以组合得到i的总组合个数
        // 先把第一个元素加入考虑范畴
        dp[nums[0]+1000] = 1;
        dp[-nums[0]+1000] += 1; // 注意这里要用+=,如果nums[0]为0的话，正负是相等的
        // 再把剩下的元素依次加入考虑范畴
        for(int i =1;i<nums.size();i++){ // 从第二元素开始考虑
            vector<int> next(2001,0);
            for(int j=0;j<=2000;j++){ // 遍历所有状态，更新状态
                if(j-nums[i]>=0 && j-nums[i]<=2000) // 表示nums[i]在表达式中取正，这里这两个边界判定不能漏
                    next[j] += dp[j-nums[i]]; // 注意这里要用+=,如果nums[i]为0的话，正负是相等的
                if(j+nums[i]>=0 && j+nums[i]<=2000) // 表示nums[i]在表达式中取负，这里这两个边界判定不能漏
                    next[j] += dp[j+nums[i]]; // 注意这里要用+=,如果nums[i]为0的话，正负是相等的
            }
            dp = next;
        }
        return dp[S+1000];
    }
};
// 时间复杂度：**O(n*sum)**  
// 空间复杂度：**O(sum)**
```

### medium[560.和为K的子数组](../Coding\数组\中等\560.和为K的子数组.md)

```C++
/*
*/
class Solution {
public:
    int subarraySum(vector<int>& nums, int k) {
        unordered_map<int,int> dp; // 前缀和为键，和出现的次数为值
        dp[0] = 1; // 和为0，出现1次，就是子数组为空
        int count = 0,pre = 0; // count表示和为k的子数组总个数，pre表示当前遍历过的元素的总和
        for(int i=0;i<nums.size();i++){
            pre += nums[i];
            if(dp.find(pre-k)!=dp.end()) // 如果前缀和pre-k存在，那么前缀和pre-k对应的右边界下标i到前缀和pre对应的右边界下标j之间的连续子数组nums[i,j]的和就为k
                count += dp[pre-k]; // 因为此时下标j是确定的，那么有多少种下标i就有多少种和为k的子数组，所以count += mp[pre - k]
            dp[pre] ++; // 将当前的前缀和pre加入到哈希表，这里直接用++，是因为不同的下标j可能有相同的前缀和pre，但是因为下标j不同，最终的子数组也不同，所有都要加到count上。
        }
        return count;
    }   
};
// 时间复杂度：**O(n)**  
// 空间复杂度：**O(n)**
```

### medium[621.任务调度器](../Coding\数组\中等\621.任务调度器.md)

```C++
/*
*/
class Solution {
public:
    int leastInterval(vector<char>& tasks, int n) {
        vector<int> cnt(26);
        for(char c:tasks) // 统计任务频数
            cnt[c-'A']++;

        int maxcount = INT_MIN;
        int maxnum = 0;
        for(int i=0;i<26;i++) // 贪心维护具有相同的最大数量(maxcount)的任务个数(maxnum)
            if(cnt[i]>maxcount){
                maxcount = cnt[i];
                maxnum = 1;
            }
            else if(cnt[i]==maxcount)
                maxnum++; // 统计最后一行的任务数

        int len = tasks.size();
        return max(len,maxnum+(maxcount-1)*(n+1)); // 在任意的情况下，需要的最少时间就是(maxcount−1)(n+1)+maxnum 和 ∣task∣ 中的较大值
                                                   // n+1代表列，(maxcount−1)代表去掉最后一个行剩下的行
    }
};
// 时间复杂度：**O(nlogn)**  
// 空间复杂度：**O(1)**
```

### medium[739.每日温度](../Coding\数组\中等\739.每日温度.md)

```C++
/*
*/
class Solution {
public:
    vector<int> dailyTemperatures(vector<int>& T) {
       int n = T.size();
       vector<int> res(n); // 默认值为0
       stack<int> s;// 单调栈存储的是下标
       for(int i=0;i<n;i++){
           while(!s.empty() && T[i] > T[s.top()]){ // 如果栈不为空并且当前元素大于栈顶元素，那么当前温度就是栈顶温度所要等待的温度
               res[s.top()] = i-s.top(); // 更新栈顶元素需要等待的天数
               s.pop(); // 栈顶元素已经计算过了，就出栈
           }
           s.push(i); // 将当前温度加入栈
       } 
       return res;
    }
};
// 时间复杂度：**O(n)**  
// 空间复杂度：**O(n)**
```

### medium[剑指-Offer-56--I.数组中数字出现的次数](../Coding\数组\中等\剑指-Offer-56--I.数组中数字出现的次数.md)

```C++
/*
*/
class Solution {
public:
    vector<int> singleNumbers(vector<int>& nums) {
        int ret = 0;
        for(int i:nums)
            ret ^=i; // 最终ret是两个只出现一次的数的异或结果
        int div = 1;
        while((ret&div)==0) // 找到ret最右边为1的那一位，两个只出现一次的数在这一位上是不同的，记为第x位
            div = (div << 1); 
        int a=0,b=0;
        for(int i:nums){ // 第x位为0的异或到a上，为1的异或到b上
            if((i&div) == 0)
                a ^= i;
            else
                b ^= i;
        }
        return vector<int>{a,b}; // 最终a和b就是只出现一次的两个数字
    }
};
// 时间复杂度：**O(n)**  
// 空间复杂度：**O(1)**
```

### medium[剑指-Offer-56--II.数组中数字出现的次数-II](../Coding\数组\中等\剑指-Offer-56--II.数组中数字出现的次数-II.md)

```C++
/*
*/
class Solution {
public:
    int singleNumber(vector<int>& nums) {
        int res = 0;
        for (int i = 0, sub = 0; i < 32; ++i, sub = 0) {
            for (auto &n : nums)
                sub += ((n >> i) & 1);
            if (sub % 3)
                res |= (1 << i);
        }
        return res;
    }
};
// 时间复杂度：**O(n)**  
// 空间复杂度：**O(1)**
```

## 12、双指针 {#customname12}

### [11.盛最多水的容器](../Coding\双指针\11.盛最多水的容器.md)

```C++
/*
*/
class Solution {
public:
    int maxArea(vector<int>& height) {
        int l=0, r=height.size()-1;
        int ans = 0; // 最大存水量
        while(l<r){
            int area = min(height[l],height[r])*(r-l); // 当前存水量
            ans = max(ans,area);
            if(height[l]<=height[r]) // 关键在于每次移动高度小的那边。
                                     // 因为移动高的不可能让水面更高，但积水宽度却变小了，总的面积一定更小
                l++;
            else
                r--;
        }
        return ans;
    }
};
// 时间复杂度：**O( n )**  
// 空间复杂度：**O( 1 )**
```

### [剑指-Offer-57--II.和为s的连续正数序列](../Coding\双指针\剑指-Offer-57--II.和为s的连续正数序列.md)

```C++
/*
*/
class Solution {
public:
    vector<vector<int>> findContinuousSequence(int target) {
        vector<vector<int>> res;
        int l=1, r=2; // 正整数从1,2开始
        while(l<r){// 类似于滑动窗口，实际上这个滑动窗口枚举的是左边界
            int sum = (l+r)*(r-l+1)/2;// 等差数列求和
            if(sum == target){
                vector<int> tmp;
                for(int i=l;i<=r;i++)
                    tmp.push_back(i);
                res.push_back(tmp);
                l++;// 移动窗口左边界，此时移动右边界不可能再等于target，只会更大
            }
            else if(sum > target)// 窗口内和大于目标值，移动左边界，小于移动右边界
                l++;
            else 
                r++;
        }
        return res;
    }
};
// 时间复杂度：**O(n)**  
// 空间复杂度：**O(1)**
```

### [面试题-10.01.合并排序的数组](../Coding\双指针\面试题-10.01.合并排序的数组.md)

```C++
/*
*/
class Solution {
public:
    void merge(vector<int>& A, int m, vector<int>& B, int n) {
        int a = m-1, b = n-1, c = m+n-1; // 倒序
        while(a<c){ // a等于c了，说明b减为0了
            if(a==-1 || A[a] < B[b])
                A[c--] = B[b--];
            else
                A[c--] = A[a--];
        }
    }
};
// 时间复杂度：**O(n)**  
// 空间复杂度：**O(1)**
```

## 13、贪心 {#customname13}

### [392.判断子序列](../Coding\贪心\392.判断子序列.md)

```C++
/*
*/
//  自己写的贪心解法 
class Solution {
public:
    bool isSubsequence(string s, string t) {
        if(s.size()==0)return true;
        if(t.size()==0)return false;
        int k=0;
        for(char T:t){
            if(T==s[k]){
                k++;
                if(k==s.size())
                    return true;
            }
        }
        return false;
    }
};
// 时间复杂度：**O(n)**  
// 空间复杂度：**O(1)**
```

### [1403.非递增顺序的最小子序列](../Coding\贪心\1403.非递增顺序的最小子序列.md)

```C++
/*
*/
class Solution {
public:
    vector<int> minSubsequence(vector<int>& nums) {
        sort(nums.begin(),nums.end(),greater<int>());//greater是一个模板函数，用于比较
        int sum = 0;
        for(int v:nums)
            sum+=v;
        int ts = 0;
        for(int i =0;i<nums.size();i++)
        {
            ts += nums[i];
            if(ts > sum-ts)
                return vector<int>(nums.begin(),nums.begin()+i+1);//这里构造不取右界，所有+1,才能取到i
        }
        return nums;//只有一个元素，或者没有元素时返回
    }
};
// 自己写的解法
class Solution {
public:
    void quicksort(vector<int>& nums,int L,int R){// 从大到小快排
        if(L>=R)// 注意不能漏
            return;
        int p = rand()%(R-L+1)+L;
        swap(nums[L],nums[p]);
        int j = R+1;
        for(int i=R;i>=L+1;i--)
            if(nums[i]<=nums[L])
                swap(nums[i],nums[--j]);
        swap(nums[L],nums[--j]);
        quicksort(nums,L,j-1);
        quicksort(nums,j+1,R);
    }
    vector<int> minSubsequence(vector<int>& nums) {
        // 降序排序
        srand((unsigned)time(nullptr));
        quicksort(nums,0,nums.size()-1);
        // 求和
        int sum = 0;
        for(int i:nums)
            sum += i;
        // 计算结果
        int tmp = 0;
        vector<int> res;
        for(int i=0;i<nums.size();i++){
            tmp += nums[i];
            res.push_back(nums[i]);
            if(tmp > sum - tmp)
                break;
        }
        return res;
    }
};
// 时间复杂度：**O(nlogn)**  
// 空间复杂度：**O(logn)**
```

### [1518.换酒问题](../Coding\贪心\1518.换酒问题.md)

```C++
/*
*/
class Solution {
public:
    // 思想是每次用ex换能多喝1瓶，每换一次，总的瓶子数就会减少ex-1，损失ex-1个瓶子，(n-ex)/(ex-1)+1为能换多少次（能损失多少次），每损失一次，就能多喝一瓶，因此(n-ex)/(ex-1)+1也就是能多喝几瓶，加上一开始喝的n即为总的能喝几瓶。
    int numWaterBottles(int n, int ex) {
        // return (n*ex-1)/(ex-1);
        // return (n-1)/(ex-1)+n;// 三种方法是等价的，可以相互推出
        return n>=ex?(n-ex)/(ex-1)+1+n:n;// +1的目的是保证(n-ex)/(ex-1) < n_exchange一定成立，因为(n-ex)/(ex-1)是整数除，会舍弃小数部分，+1就一定能满足这个不等式。或者可以理解为(n-ex)里减去的ex 和 (n-ex)/(ex-1)结果的小数部分加在一起，一定能再换一瓶。
    }                               // 注意这个+1的目的
};
// 时间复杂度：**O(1)**  
// 空间复杂度：**O(1)**
```

## 14、图 {#customname14}

### [剑指](../)

## 15、位运算 {#customname15}

### [371.两整数之和](../Coding\位运算\371.两整数之和.md)

```C++
/*
*/
class Solution {
public:
    int getSum(int a, int b) {
        while(b!=0){ // 进位不为0，就一直加，这里一定要是不等于0，才能处理负数
            int tmp = a^b; // 异或得到无进位加法结果
            b = (unsigned)(a&b)<<1; // 相与并左移一位得到进位结果
            a = tmp; // a赋值为无进位加法结果
        }
        return a; // 当while退出时，进位b为0，那么上一次的无进位加法结果就是最终的和，也就是a
    }
};
// 时间复杂度：**O(n)**  
// 空间复杂度：**O(1)**
```

### [1356.根据数字二进制下1的数目排序](../Coding\位运算\1356.根据数字二进制下1的数目排序.md)

```C++
/*
*/
// sort函数结合lambda表达式，(1710. 卡车上的最大单元数)也用过
// 自己的解法
class Solution {
public:
    void quicksort(vector<int>& nums,int L,int R,vector<int>& bit){
        if(L>=R)        // 递归终止条件不能忘
            return;
        int p = rand()%(R-L+1)+L;
        swap(nums[R],nums[p]);
        int i = L-1;
        for(int j=L;j<=R-1;j++)
            if(bit[nums[j]]<bit[nums[R]] || (bit[nums[j]]==bit[nums[R]] && nums[j]<=nums[R] ))// 按二进制中1的个数升序排序，个数相同按数值升序
                swap(nums[++i],nums[j]);
        swap(nums[++i],nums[R]);
        quicksort(nums,L,i-1,bit);
        quicksort(nums,i+1,R,bit);
    }
    vector<int> sortByBits(vector<int>& arr) {
        vector<int> bit(10001,0); // 值域为[0,10000],利用数组下标来哈希，数组中的每个位置存下标二进制中1的个数
        for(int i=1;i<=10000;i++)
            bit[i] = bit[i>>1] + (i&1); //递推求解，i右移一位后的1的个数在之前已经被求出，再加上i的最后一位，得到i的二进制中1的个数
        // srand((unsigned)time(nullptr));
        quicksort(arr,0,arr.size()-1,bit);
        return arr;
    }
};
// 时间复杂度：**O( nlogn )**  
// 空间复杂度：**O( n )**
```

### [面试题-05.03.-翻转数位](../Coding\位运算\面试题-05.03.-翻转数位.md)

```C++
/*
*/
class Solution{
public:
    // 思路很好，特别是l=r+1
    int reverseBits(int num) {
        int l=0,r=0,Max=0; // 以0为分界点，L是0左边连续1的数量+1（翻转的0），R是0右边连续1的数量，Max记录最大值
        for(int i=0;i<32;i++){
            if((num&1)==1)
                r++;
            else{ // 注意理解这里，遇到0的处理方式
                l = r+1; // 当遇见0时， 0的左边连续1的数量等于上一个0右边连续1的数量加一（当前0本身反转后算一个长度）
                r = 0;
            }
            Max = max(l+r,Max);
            num >>= 1;
        }
        return Max;
    }
};
// 时间复杂度：**O(n)**  
// 空间复杂度：**O(1)**
```

### [面试题-17.04.-消失的数字](../Coding\位运算\面试题-17.04.-消失的数字.md)

```C++
/*
*/
class Solution {
public:
    int missingNumber(vector<int>& nums) {
        int n = nums.size();
        int tmp = 0;
        for(int i=0;i<nums.size();++i)
            tmp ^= i^nums[i];
        return n^tmp;
    }
};
// 时间复杂度：**O(n)**  
// 空间复杂度：**O(1)**
```

## 16、栈 {#customname16}

### [155.最小栈](../Coding\栈\155.最小栈.md)

```C++
/*
*/
class MinStack {
    stack<int> x_stack;
    stack<int> min_stack;// 存储最小值的辅助栈，栈顶元素为目前为止x_stack中最小元素
public:
    MinStack() {
        min_stack.push(INT_MAX);
    }
    
    void push(int x) {
        x_stack.push(x);
        min_stack.push(min(min_stack.top(), x));// 入栈一个考虑x后的最小值
    }
    
    void pop() {
        x_stack.pop();
        min_stack.pop();
    }
    
    int top() {
        return x_stack.top();
    }
    
    int getMin() {
        return min_stack.top();
    }
};
```

### [225.用队列实现栈](../Coding\栈\225.用队列实现栈.md)

```C++
/*
*/
class MyStack225 {
public:
    queue<int> q;

    /** Initialize your data structure here. */
    MyStack225() {

    }

    /** Push element x onto stack. */
    void push(int x) {
        int n = q.size();
        q.push(x);
        for (int i = 0; i < n; i++) {
            q.push(q.front());
            q.pop();
        }
    }
    
    /** Removes the element on top of the stack and returns that element. */
    int pop() {
        int r = q.front();
        q.pop();
        return r;
    }
    
    /** Get the top element. */
    int top() {
        int r = q.front();
        return r;
    }
    
    /** Returns whether the stack is empty. */
    bool empty() {
        return q.empty();
    }
};
```

### [232.用栈实现队列](../Coding\栈\232.用栈实现队列.md)

```C++
/*
*/
class MyQueue {// 双栈
public:
    stack<int> s1,s2;
    /** Initialize your data structure here. */
    MyQueue() {
        
    }
    
    /** Push element x to the back of queue. */
    void push(int x) {
        s1.push(x);
    }
    
    /** Removes the element from in front of queue and returns that element. */
    int pop() {
        if(s2.empty())
        {
            while(!s1.empty())
            {
                int number = s1.top();
                s2.push(number);
                s1.pop();
            }
        }
        int i = s2.top();
        s2.pop();
        return i;
    }
    
    /** Get the front element. */
    int peek() {
        if(s2.empty())
        {
            while(!s1.empty())
            {
                int number = s1.top();
                s2.push(number);
                s1.pop();
            }
        }
        return s2.top();
    }
    
    /** Returns whether the queue is empty. */
    bool empty() {
        return s1.empty()&&s2.empty();
    }
};
```

### [496.下一个更大的元素I](../Coding\栈\496.下一个更大的元素I.md)

```C++
/*
*/
// 自己写的解法
class Solution {
public: // 单调栈，从后往前遍历构造
    vector<int> nextGreaterElement(vector<int>& nums1, vector<int>& nums2) {
        unordered_map<int,int> mp;
        stack<int> S;
        for(int i=nums2.size()-1; i>=0; i--){ // 构建单调栈
            while(!S.empty() && S.top()<=nums2[i])
                S.pop();
            mp[nums2[i]] = S.empty()?-1:S.top();
            S.push(nums2[i]);
        }
        vector<int> res;
        for(int i:nums1) // 获取结果
            res.push_back(mp[i]);
        return res;
    }
};
```

### [剑指Offer-31.栈的压入弹出序列](../Coding\栈\剑指Offer-31.栈的压入弹出序列.md)

```C++
/*
*/
class Solution31 {
public:
    bool validateStackSequences(vector<int>& pushed, vector<int>& popped) {
        if(pushed.size() != popped.size()) return false;
        stack<int> stk;
        int i = 0;
        for(auto x : pushed){
            stk.push(x);
            while(!stk.empty() && stk.top() == popped[i]){
                stk.pop();
                i++;
            }
        }
        return stk.empty();
    }
};
```

## 17、字符串 {#customname17}

### [3.无重复字符的最长子串](../Coding\字符串\3.无重复字符的最长子串.md)

```C++
/*
*/
class Solution {
public:
    int lengthOfLongestSubstring(string s) {
        unordered_set<char> sets;   // 哈希集合，记录每个字符是否出现过
        int n = s.size();
        int l=0;    // l为右指针，初始值为0，相当于我们在字符串的左边界，还没有开始移动
        int ans = 0;// 初始长度为0，s为空时，直接返回0
        for(int r=0;r<n;r++){   // 枚举所有滑动窗口的新的右端点
            while(l<r && sets.find(s[r])!=sets.end())//如果右端点的值已经存在集合中，就右移左端点并丢弃左边界元素，直到新的右端点的值不在集合中
                sets.erase(s[l++]); // 左指针向右移动一格，移除一个字符
            sets.insert(s[r]);// 将右端点的值加入集合，此时窗口为以新右端点为右边界的最大窗口
            ans = max(ans,r-l+1);// 更新最大长度
        }
        return ans;
    }
};
// 时间复杂度：**O(n)**  
// 空间复杂度：**O(n)**
```

### [5.最长回文子串](../Coding\字符串\5.最长回文子串.md)

```C++
/*
*/
class Solution { // 中心扩展法 + 贪心
public:
    pair<int,int> expend1(string& s,int L,int R){
        while(L>=0 && R<=s.size()-1 && s[L]==s[R]){// 注意边界条件
            L--;
            R++;
        }
        return {L+1,R-1};
    }
    string longestPalindrome(string s) {
        int start=0,end=0;
        for(int i=0;i<s.size();i++){// 枚举回文中心
            pair<int,int> p1 = expend1(s,i,i); // 回文中心是一个字符，回文串长度为奇数
            pair<int,int> p2 = expend1(s,i,i+1); // 回文中心是两个字符，回文串长度为偶数
            if(p1.second-p1.first > end-start){// 注意是R-L，不能反，更新最长长度
                start = p1.first;
                end   = p1.second;
            }
            if(p2.second-p2.first > end-start){
                start = p2.first;
                end   = p2.second;
            }
        }
        return s.substr(start,end-start+1);

    }
};
// time：O(n²)
// space：O(1)
```

### [17.电话号码的字母组合](../Coding\字符串\17.电话号码的字母组合.md)

```C++
/*
*/
// 自己写的解法
class Solution {
public:
    unordered_map<char,string> mp{
            {'2', "abc"},
            {'3', "def"},
            {'4', "ghi"},
            {'5', "jkl"},
            {'6', "mno"},
            {'7', "pqrs"},
            {'8', "tuv"},
            {'9', "wxyz"}};
    string digit;
    void backtrack(vector<string>& ans,string& tmp,int idx){
        if(idx == digit.size())
            ans.push_back(tmp);
        else{
            for(char i:mp[digit[idx]]){
                tmp.push_back(i);
                backtrack(ans,tmp,idx+1);
                tmp.pop_back();
            }
        }
    }
    vector<string> letterCombinations(string digits) {
        vector<string> ans;
        if(digits.empty())
            return ans;
        string tmp; // 存储回溯结果
        digit = digits; // 用于遍历
        backtrack(ans,tmp,0); // 0是digit里的下标
        return ans;
    }
};
// 时间复杂度：**O(O(3m次方+4n次方))**  
// 空间复杂度：**O(m+n)**
```

### [22.括号生成](../Coding\字符串\22.括号生成.md)

```C++
/*
*/
class Solution {
public:
    void backward(vector<string>&res,string& tmp,int open,int close,int n){
        // open、close分别为已有序列左右括号数量，n为要生成括号的对数
        if(tmp.size() == 2*n){  // 找到一个完整有效的排列
            res.push_back(tmp);
            return;
        }
        
        // 任意时候都可以添加左括号，但只有在添加过的左括号数大于添加过的右括号数时才可以添加右括号
        if(open < n){   // 可以添加左括号就添加尝试一下
            tmp.push_back('(');
            backward(res,tmp,open+1,close,n);
            tmp.pop_back(); // 关键
        }
        if(open > close){   // 可以添加右括号就添加尝试一下
            tmp.push_back(')');
            backward(res,tmp,open,close+1,n);
            tmp.pop_back(); // 关键
        }
    }
    vector<string> generateParenthesis(int n) {
        vector<string> res;// 结果向量
        string tmp;// 用于存储当前枚举排列的字符串
        backward(res,tmp,0,0,n);
        return res;
    }
};
// 时间复杂度：**O(4ⁿ/√n)**  
// 空间复杂度：**O(n)**
```

### [49.字母异位词分组](../Coding\字符串\49.字母异位词分组.md)

```C++
/*
*/
class Solution {
public:
    vector<vector<string>> groupAnagrams(vector<string>& strs) {
        unordered_map<string,vector<string>> mp;
        for(string i:strs){
            vector<int> cnt(26);
            for(char s:i) // 统计字符串i中的字符个数
                cnt[s-'a']++;
            string key;
            for(int i=0;i<26;i++) // 根据统计结果(cnt)构造哈希的key
                // key += (char)('a'+i) + to_string(cnt[i]);// 注意这里构造字符串的方法
                key += to_string('a'+i) + to_string(cnt[i]);// 注意这里构造字符串的方法，'a'+i被提升为int
            mp[key].push_back(i); // 字符串i添加到对应的key
        }
        vector<vector<string>> res;
        for(pair<string,vector<string>> p:mp) // map中值放入结果
            res.push_back(p.second);
        return res;
    }
};
// 时间复杂度：**O(n(k+|Σ|))**  
// 空间复杂度：**O(n(k+|Σ|))**
```

### [72.编辑距离](../Coding\字符串\72.编辑距离.md)

```C++
/*
*/
class Solution {
public:
    int minDistance(string word1, string word2) {
        int m = word1.size();
        int n = word2.size();
        if(m==0)return n;
        if(n==0)return m;
        vector<vector<int>> dp(m+1,vector<int>(n+1,0)); // dp[i][j] 表示 A 的前 i 个字母和 B 的前 j 个字母之间的编辑距离
        // 边界条件
        for(int i=0;i<=m;i++)
            dp[i][0] = i;
        for(int j=0;j<=n;j++)
            dp[0][j] = j;
        
        for(int i=1;i<=m;i++)
            for(int j=1;j<=n;j++)
                if(word1[i-1] == word2[j-1]) // 注意这里是-1，
                    // dp[i-1][j]+1 表示删除word1中的第i个字符
                    // dp[i][j-1]+1 表示删除word2中的第j个字符
                    dp[i][j] = min(dp[i-1][j]+1,min(dp[i][j-1]+1,dp[i-1][j-1])); // 如果最后两个字符相等
                else
                    // dp[i-1][j-1]+1 表示将word1中的第i个字符替换为word2中的第j个字符
                    dp[i][j] = min(dp[i-1][j]+1,min(dp[i][j-1]+1,dp[i-1][j-1]+1)); // 如果最后两个字符不相等
        return dp[m][n];
    }
};
// 时间复杂度：**O(mn)**  
// 空间复杂度：**O(mn)**
```

### [139.单词拆分](../Coding\字符串\139.单词拆分.md)

```C++
/*
*/
class Solution {
public:
    bool wordBreak(string s, vector<string>& wordDict) {
        // 单词加入集合，并记录最大长度
        unordered_set<string> sets;
        int maxlen = 0;
        for(string i:wordDict){
            sets.insert(i);
            maxlen = max(maxlen,(int)i.size());
        }
        // dp[i] 表示字符串 s 前 i 个字符组成的字符串 s[0..i-1] 是否能被空格拆分成若干个字典中出现的单词
        vector<int> dp(s.size()+1,0);
        dp[0] = 1;
        for(int i=1;i<=s.size();i++)// 枚举所有状态
            for(int j=i;j>=0 && j>=i-maxlen;j--) // 从dp[0]、dp[1]、...、dp[i-1]（不一定全用，只要发现可以转移，dp[i]就会被设为true）转移到dp[i]
                // 这里倒序枚举分割点，分割点如果将最后一个单词分割得超过了字典中的最长单词，必然就没必要继续枚举下去了
                if(dp[j] && sets.find(s.substr(j,i-j))!=sets.end()){ // 如果单词s.substr(j, i - j)存在于字典中，并且dp[j]为true，表示s[0,i]可以拆分，dp[i]设为true
                    dp[i] = 1;
                    break;
                }
        return dp[s.size()];
    }
};
// 时间复杂度：**O( n² )**  
// 空间复杂度：**O( n )**
```

### [394.字符串解码](../Coding\字符串\394.字符串解码.md)

```C++
/*
*/
class Solution {
public:
    string getDigits(string &s, size_t &ptr) {// 解析数字
        string ret = "";
        while (isdigit(s[ptr])) {// 使用while的目的是取完连续的数字，"345"表示345
            ret.push_back(s[ptr++]);
        }
        return ret;// 返回形如"345"的字符串
    }

    string getString(vector <string> &v) {// 将字符串向量中的所有元素组合起来成为一整个字符串
        string ret;
        for (const auto &s: v) {
            ret += s;
        }
        return ret;
    }

    string decodeString(string s) {
        vector <string> stk;// 辅助栈，实质上是辅助向量
        size_t ptr = 0;// 当前所处理字符的下标

        while (ptr < s.size()) {// 如果还没有处理完所有字符，就循环
            char cur = s[ptr];// 取当前字符
            if (isdigit(cur)) {// 如果当前字符是数字
                // 获取一个数字并进栈
                string digits = getDigits(s, ptr);
                stk.push_back(digits);
            } else if (isalpha(cur) || cur == '[') {// 如果当前字符是字母或者左中括号
                // 获取一个字母并进栈
                stk.push_back(string(1, s[ptr++])); 
            } else {// 如果当前字符是右中括号
                ++ptr;// 跳过右中括号
                vector <string> sub;// 存储子串的字符串向量
                while (stk.back() != "[") {// 一直出栈直到左中括号，出栈同时添加到字符串向量中
                    sub.push_back(stk.back());
                    stk.pop_back();
                }
                reverse(sub.begin(), sub.end());// 反转
                // 左括号出栈
                stk.pop_back();// 跳过左中括号
                // 此时栈顶为当前 sub 对应的字符串应该出现的次数
                int repTime = stoi(stk.back()); 
                stk.pop_back();
                string t, o = getString(sub);
                // 构造字符串
                while (repTime--) t += o; 
                // 将构造好的字符串入栈
                stk.push_back(t);
            }
        }

        return getString(stk);
    }
};
// 时间复杂度：**O(S+∣s∣)**  
// 空间复杂度：**O(S)**
```

### [395.至少有K个重复字符的最长子串](../Coding\字符串\395.至少有K个重复字符的最长子串.md)

```C++
/*
*/
class Solution {
public:
    int longestSubstring(string s, int k) {
        int ret = 0; // 满足要求的最长子串的长度
        int n =s.size();
        for(int i=1;i<26;i++){  // 枚举滑动窗口内的字符种类数目，枚举最长子串中的字符种类数目。这个是重点，先固定字符种类数，然后再找k个重复
            int l=0;        // 滑动窗口的左右边界下标
            int r=0;
            vector<int> cnt(26);    // 滑动窗口内部每个字符出现的次数
            int total=0;    // 滑动窗口内的字符种类数目
            int less=0;     // 滑动窗口中出现次数小于k的字符数量，比如为2，表示有两种字符出现次数小于k

            while(r<n){         // 对于每一种字符种类数目限定，枚举滑动窗口的右边界下标
                cnt[s[r]-'a']++;
                if(cnt[s[r]-'a']==1){
                    total++;
                    less++;
                }
                if(cnt[s[r]-'a']==k){
                    less--;
                }

                while(total>i){     // 对于每一个右边界下标，不断右移左边界下标使得窗口内的字符种类数目为限定的数目
                    cnt[s[l]-'a']--;
                    if(cnt[s[l]-'a']==k-1){
                        less++;
                    }
                    if(cnt[s[l]-'a']==0){
                        total--;
                        less--;
                    }
                    l++;// 窗口左边界右移
                }
                // 找到一个以r为右边界，l为左边界，且包含字符种类数目为i个窗口
                if(less==0)         // 如果这个窗口中数量小于k的字符种类数目为0，那么就找到了一个窗口，
                                    // 窗口中每种字符出现的次数都不小于k，就更新最大的窗口长度
                    ret = max(ret,r-l+1);
                r++;   // 窗口右边界右移
            }
        }
        return ret;
    }
};
```

### [438.找到字符串中所有字母异位词](../Coding\字符串\438.找到字符串中所有字母异位词.md)

```C++
/*
*/
class Solution {
public:
    bool check(int s_[],int p_[]){  //判断两个单词是否相同
        for(int i=0;i<26;i++)
            if(s_[i]!=p_[i])
                return false;
        return true;
    }
    vector<int> findAnagrams(string s, string p) {
        vector<int> res;
        if(p.size()>s.size())
            return res;
        // 采用数组代替哈希表，速度更快，但是要自己写判断函数
        int s_[26] = {0}; // 列表解析，每个元素都设为0
        int p_[26] = {0};
        // 统计p的字母
        for(int i=0;i<p.size();i++){
            s_[s[i]-'a']++;
            p_[p[i]-'a']++;
        }
        // 判断初始窗口是否满足要求
        if(check(s_,p_))
            res.push_back(0);
        // 指向窗口左右边界的双指针
        int l=0;
        int r=p.size()-1;

        // 滑动窗口
        while(r<s.size()-1){
            s_[s[++r]-'a']++; // 减一个字符
            s_[s[l++]-'a']--; // 加一个字符
            if(check(s_,p_)) // 判断是否是字母异位词
                res.push_back(l);
        }
        return res;
    }
};
// 时间复杂度：**O( np )**  
// 空间复杂度：**O( p )**
```

### [459.重复的子字符串](../Coding\字符串\459.重复的子字符串.md)

```C++
/*
*/
class Solution {
public:
    void get_match(string& T,int match[]){
        match[0] = -1;
        for(int i=1;i<T.size();i++){
            int p = match[i-1];
            while(p!=-1 && T[p+1]!=T[i])
                p = match[p];
            if(T[p+1]==T[i])
                match[i] = p+1;
            else
                match[i] = -1;
        }
    }
    int KMP(string& S,string& T){
        int s=0;
        int t=0;
        int match[T.size()];
        get_match(T,match);
        while(s<S.size() && t<T.size()){
            if(S[s]==T[t]){
                s++;
                t++;
            }
            else if(t>0)
                t = match[t-1]+1;
            else
                s++;
        }
        return t==T.size() ? (s-T.size()):-1;
    }
    bool repeatedSubstringPattern(string s) {
        string res = s.substr(1)+s.substr(0,s.size()-1);// 这里注意是s+s，然后去掉头尾两个字符
        return KMP(res,s)!=-1;
    }
};
```

### [516.最长回文子序列](../Coding\字符串\516.最长回文子序列.md)

```C++
/*
*/
class Solution {
public:
    int longestPalindromeSubseq(string s) {
        int n = s.size();
        // dp[i][j] 表示s的第i个字符到第j个字符组成的子串中，最长的回文序列长度
        vector<vector<int>> dp(n,vector<int>(n));
        for(int i=n-1;i>=0;i--){ // i从最后一个字符往前遍历，作为区间左端点
            dp[i][i] = 1; // 第i个字符是长度为1的回文子串
            for(int j=i+1;j<n;j++) // j从i+1开始往后遍历，作为区间右端点
                if(s[i]==s[j])
                    dp[i][j] = dp[i+1][j-1] + 2;
                else
                    dp[i][j] = max(dp[i+1][j],dp[i][j-1]);// 取长度比[i,j]小1的区间的最大值
        }
        return dp[0][n-1];
    }
};
// 时间复杂度：**O( n² )**  
// 空间复杂度：**O( n² )**
```

### [647.回文子串](../Coding\字符串\647.回文子串.md)

```C++
/*
*/
class Solution {
public:
    int count = 0;
    void expend(string& s,int L,int R){
        while(L>=0 && R<=s.size()-1 && s[L]==s[R]){
            L--;
            R++;
            count++;// 注意这里，每扩一次就+1
        }
    }
    int countSubstrings(string s) {
        for(int i=0;i<s.size();i++){
            expend(s,i,i); // 长度为奇数
            expend(s,i,i+1); // 长度为偶数
        }
        return count;
    }
};
// 时间复杂度：**O(n)**  
// 空间复杂度：**O(1)**
```

### [680.验证回文字符串-Ⅱ](../Coding\字符串\680.验证回文字符串-Ⅱ.md)

```C++
/*
*/
class Solution {
public:
    bool check(string& s,int L,int R){
        while(L<R){
            if(s[L]!=s[R])
                return false;
            L++;
            R--;
        }
        return true;
    }
    bool validPalindrome(string s) { // 模拟删除一个字符
        int L=0,R=s.size()-1;
        while(L<R){
            if(s[L]==s[R]){
                L++;
                R--;
            }
            else    // 这个else只被执行一次，也就是最多删除一个字符
                    // volidsubstr(s,start+1,end)为删除下标为start的字符
                    // volidsubstr(s,start,end-1)为删除下标为end的字符
                return check(s,L,R-1) || check(s,L+1,R); //采用这种方式能考虑处理多种情况
        }
        return true;
    }
};
// 时间复杂度：**O(n)**  
// 空间复杂度：**O(1)**
```

### [788.旋转数字](../Coding\字符串\788.旋转数字.md)

```C++
/*
*/
class Solution {
public:
    int rotatedDigits(int N) {
        int count = 0;
        vector<int> dp(N+1,0);// dp[i]表示数字i是否是好数，为2表示是，为1表示不是
        for(int i=1;i<=N;i++){
            if(i==3||i==4||i==7||
                dp[i%10]==1||dp[i/10]==1){ // 或者个位不是好数，或者去掉个位后不是好数
                    dp[i]=1;
                }
            else if(i==2||i==5||i==6||i==9||// 进入这个else if时i就每一位都不是3,4,7，只能是1,2,3,6,8,9其中之一
                dp[i%10]==2||dp[i/10]==2){ // 或者个位是好数，或者去掉个位后是好数
                    dp[i]=2;
                    count++;
                }
        }
        return count;
    }
};
```

### [1071.字符串的最大公因子](../Coding\字符串\1071.字符串的最大公因子.md)

```C++
/*
*/
class Solution {
public:
    string gcdOfStrings(string str1, string str2) {
        return (str1 + str2 == str2 + str1)  ?  str1.substr(0, __gcd(str1.size(), str2.size()))  : "";
    }
};
```

### [剑指-Offer-38.字符串的排列](../Coding\字符串\剑指-Offer-38.字符串的排列.md)

```C++
/*
*/
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

### [剑指-Offer-67.把字符串转换成整数](../Coding\字符串\剑指-Offer-67.把字符串转换成整数.md)

```C++
/*
*/
class Solution {
public:
    int strToInt(string str) {
        int i = 0, flag = 1; // 默认flag = 1，正数
        long res = 0;
        while (str[i] == ' ') i++;
        if (str[i] == '-') flag = -1;
        if (str[i] == '-' || str[i] == '+') i++;
        for (; i < str.size() && isdigit(str[i]); i ++)  {
            res = res * 10 + (str[i] - '0');
            if (res >= INT_MAX && flag == 1) return  INT_MAX;
            if (res > INT_MAX && flag == -1) return  INT_MIN;
        } 
        return flag * res;
    }
};
// 时间复杂度：**O(n)**  
// 空间复杂度：**O(1)**
```
