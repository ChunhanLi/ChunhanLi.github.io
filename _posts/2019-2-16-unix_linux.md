---
layout:     post
title:      unix_linux
subtitle:   unix_linux
date:       2019-2-16
author:     Midone
header-img: img/post-bg-re-vs-ng2.jpg
catalog: false
tags:
    - 编程语言
---

## Unix/Linux操作指令

课程网址： https://swcarpentry.github.io/shell-novice/


### 1. Introducing the shell

https://swcarpentry.github.io/shell-novice/01-intro/index.html

### 2. Navigating Files and Directories

https://swcarpentry.github.io/shell-novice/02-filedir/index.html

```
pwd ## print working directory

ls
ls -l ### long lisst
ls -l -h / ls -lh### human readable KB/MB
ls -a ## show all
cd - ## back to last channel
cd ..
cd ~
```


### 4. Pipes and Filters
https://swcarpentry.github.io/shell-novice/04-pipefilter/index.html
```{bash}
wc # word count left to right: lines words characters
wc -l # only lines -w words - c character
> # tells the shell to redirect the command’s output to a file instead of printing it to the screen.
>> ## appends the string to the file if it already exists
cat ### stands for “concatenate”: it prints the contents of files one after another.

###This displays a screenful of the file, and then stops. You can go forward one screenful by pressing the spacebar, or back one by pressing b. Press q to quit.
less filename

sort -n ### numerical instead of alphanumerical sort
head -n 1
tail
echo
| # pipe

# wc -l notes.txt
# wc -l < notes.txt

# < is used to redirect input to a command.

# In both examples, the shell returns the number of lines from the input to the wc command. In the first example, the input is the file notes.txt and the file name is given in the output from the wc command. 

# In the second example, the contents of the file notes.txt are redirected to standard input. It is as if we have entered the contents of the file by typing at the prompt. Hence the file name is not given in the output - just the number of lines.

uniq #The command uniq removes adjacent duplicated lines from its input.
sort salmon.txt | uniq ## remove all the duplicate lines

uniq -c ### prefix lines by the number of occurrences

cut -d , -f 2 animals.txt # uses the -d flag to separate each line by comma, and the -f flag to print the second field in each line, to give the following output:


```



### STA 141C

```
tr ###替换
nl ### number lines of files
echo -e ###可以换行
grep 'bicycle' xxx.csv > aa.txt

sed -i 'nd' filename ### '1d'删除第一行


awk NF filename #NF代表当前行的字段数，空行的话字段数为0,被awk解释为假，因此不进行输出。

awk  '$7!=""' file > final_output ##去除第七行为0的

单引号 双引号（参数替换/命令替换）的 区别 https://blog.csdn.net/qq_40491569/article/details/83688652

awk -F "," '$'"${a}"'!=""' test.csv | wc -l ## 引用外部变量 '"${a}"'

sort -key=1,1 ###根据第一列sort 如果用key=1 会从第一列开始到最后 都参与sort
```