#### 10.2

(a)

特征值：
$$
\det(\lambda I-A)=\lambda^2+w^2=0\Rightarrow \lambda =\pm iw
$$
resolvent：
$$
\begin{aligned}
(sI-A)^{-1}
&=\left[\begin{array}{cc}{s} & {-\omega} 
\\ {\omega} & {s}\end{array}\right]\\
&=\left[
\begin{array}{cc}{\frac s{s^2+w^2}} & \frac \omega{s^2+w^2} \\ 
-\frac \omega{s^2+w^2} & {\frac s{s^2+w^2}}\end{array}\right]
\end{aligned}
$$
状态转移矩阵：
$$
\begin{aligned}
\Phi(t)&=\mathcal{L}^{-1}\left((s I-A)^{-1}\right)\\
&=\left[\begin{array}{rr}{\cos wt} & {\sin wt} \\ {-\sin wt} & {\cos wt}\end{array}\right]
\end{aligned}
$$
所以
$$
x(t)=\left[\begin{array}{rr}{\cos wt} & {\sin wt} \\ {-\sin wt} & {\cos wt}\end{array}\right]x(0)
$$
(b)略过

(c)因为$\left[\begin{array}{rr}{\cos wt} & {\sin wt} \\ {-\sin wt} & {\cos wt}\end{array}\right]$是正交矩阵，所以结论成立。

(d)
$$
\begin{aligned}
\frac d{dt}\| x(t)\|^2
&=\frac d{dt}\left( x(t)^Tx(t) \right)\\
&=2\dot x(t)^T x(t)\\
&= x(t)^T \left[\begin{array}{cc}{0} & {-\omega} \\ {\omega} & {0}\end{array}\right]x(t)\\
&=0
\end{aligned}
$$
所以
$$
\dot x(t)^T x(t) =0
$$
结论成立。



#### 10.3

(a)
$$
\begin{aligned}
e^{A}e^{B}
&=\left(\sum_{i=0}^{\infty} \frac {A^i}{i!} \right)
\left(\sum_{j=0}^{\infty} \frac {B^j}{j!} \right)\\
&=\sum_{k=0}^{\infty} \sum_{i+j=k}\frac 1 {i! j!} A^iB^j\\
&=\sum_{k=0}^{\infty}\frac 1{k!}\sum_{i+j=k}\frac {k!} {i! j!} A^iB^j\\
&=\sum_{k=0}^{\infty}\frac 1{k!}(A+B)^k & 由AB=BA\\
&=e^{(A+B)}
\end{aligned}
$$
(b)
$$
\begin{aligned}
\frac{d}{d t} e^{A t}
&=\frac{d}{d t}\left(\sum_{i=0}^{\infty} \frac {(At)^i}{i!} \right)\\
&=\sum_{i=1}^{\infty}  \frac {A^it^{i-1}}{(i-1)!}\\
&=A\sum_{i=1}^{\infty} \frac {(At)^{i-1}}{(i-1)!}\\
&=Ae^{At}\\
&=\left(\sum_{i=1}^{\infty} \frac {(At)^{i-1}}{(i-1)!}  \right) A\\
&=e^{At} A
\end{aligned}
$$



#### 10.4

(a)因为
$$
A^2=\left[\begin{array}{ll}{-1} & {1} \\ {-1} & {1}\end{array}\right]\left[\begin{array}{ll}{-1} & {1} \\ {-1} & {1}\end{array}\right]=0
$$
所以
$$
\begin{aligned}
e^{tA}
&=\sum_{i=0}^{\infty} \frac {(tA)^i}{i!} \\
&=\left[\begin{matrix}
1& 0\\
0&1
\end{matrix}\right] +\left[\begin{array}{ll}{-1} & {1} \\ {-1} & {1}\end{array}\right]t\\
&=\left[\begin{array}{ll}1-t & {t} \\ {-t} & {1+t}\end{array}\right]
\end{aligned}
$$
取$t=1$得到
$$
\begin{aligned}
e^{A}
&=\left[\begin{array}{ll}0 & {1} \\ {-1} & {2}\end{array}\right]
\end{aligned}
$$
(b)由结论可得
$$
\begin{aligned}
x(t)
&=e^{tA}x(0) \\
&=\left[\begin{array}{ll}1-t & {t} \\ {-t} & {1+t}\end{array}\right]
\left[\begin{matrix}
1\\
a
\end{matrix}\right] \\
\end{aligned}
$$
令$t=1$得到
$$
\begin{aligned}
x(1)
&=\left[\begin{array}{ll}0 & {1} \\ {-1} & {2}\end{array}\right]
\left[\begin{matrix}
1\\
a
\end{matrix}\right] \\
&=\left[\begin{matrix}
a \\
2a-1
\end{matrix}\right]
\end{aligned}
$$
由题意可得
$$
2a-1=2\Rightarrow a=\frac 3 2
$$
因此
$$
\begin{aligned}
x(t)
&=\left[\begin{array}{ll}1-t & {t} \\ {-t} & {1+t}\end{array}\right]
\left[\begin{matrix}
1\\
\frac 3 2
\end{matrix}\right] \\
&=\left[\begin{matrix}
 1+\frac 12 t\\
 \frac 3 2 +\frac 1 2t
\end{matrix}\right]\\
x(2)
&=\left[\begin{matrix}
2 \\
\frac 5 2
\end{matrix}\right]
\end{aligned}
$$



### 补充题

#### 1

求导可得
$$
\begin{aligned}
\dot x(t)
&=a(t) \exp \left(\int_{0}^{t} a(\tau) d \tau\right) x(0)\\
&=a(t)x(t)
\end{aligned}
$$
作为反例，考虑（参考自解答）
$$
A(t)=\left\{\begin{array}{ll}{A_{1}} & {0 \leq t<1} \\ {A_{2}} & {t \geq 1}\end{array}\right.
$$
那么
$$
A_{1}=\left[\begin{array}{ll}{0} & {1} \\ {0} & {0}\end{array}\right], \qquad A_{2}=\left[\begin{array}{ll}{0} & {0} \\ {1} & {0}\end{array}\right]
$$
那么
$$
\begin{aligned} x(2) &=\left(\exp A_{2}\right)\left(\exp A_{1}\right) x(0) \\ &=\left[\begin{array}{cc}{1} & {1} \\ {0} & {1}\end{array}\right]\left[\begin{array}{cc}{1} & {0} \\ {1} & {1}\end{array}\right] x(0) \\ &=\left[\begin{array}{cc}{1} & {1} \\ {1} & {2}\end{array}\right] x(0) \end{aligned}
$$
但是，在上述公式中
$$
\begin{aligned}
\int_0^2 A(t) dt
&=\int_0^1 A_1 dt +\int_1^2 A_2 dt\\
&=\left[\begin{array}{ll}{0} & {1} \\ {1} & {0}\end{array}\right]\\
&\triangleq B
\end{aligned}
$$
不难验证
$$
B^{2k}=I,B^{2k+1}= B
$$
所以
$$
\begin{aligned}
\exp(tB)
&=\sum_{i=0}^{\infty} \frac{B^i}{i!}\\
&=\sum_{k=0}^{\infty} \frac{B^{2k}}{(2k)!}+
\sum_{k=0}^{\infty} \frac{B^{2k+1}}{(2k+1)!}\\
&=I\sum_{k=0}^{\infty} \frac{1}{(2k)!}
+B \sum_{k=0}^{\infty} \frac{1}{(2k+1)!}\\
&= I\frac {e+e^{-1}} 2 +B\frac {e-e^{-1}} 2\\
&=\left[\begin{array}{cc}{1.5431} & {1.1752} \\ {1.1752} & {1.5431}\end{array}\right]
\end{aligned}
$$
因此
$$
x(2)=\left[\begin{array}{cc}{1.5431} & {1.1752} \\ {1.1752} & {1.5431}\end{array}\right]x(0)
$$

所以上述事实对高维情形不成立。