#!/usr/bin/env python
# coding: utf-8

# # LaTeX
# 
# **References:**
# 
# - http://en.wikibooks.org/wiki/LaTeX/Mathematics
# - https://stat.duke.edu/resources/computing/latex
# - http://www.equationsheet.com/
# 
# LaTeX is a large typeset system for scientific documentation which symbols for mathematics, statistics, physics, quantum mechanics, and computer science. 
# It is beyond the scope of this tutorial to cover everything, but we will go over the basics of writing high quality mathematical equations using LaTeX.
# 
# You can use LaTeX in line by like this $y = x^2$ or in block like this $$y = x^2$$
# 
# Code:
# 
#     markdown
#     You can use LaTeX in line by like this $y = x^2$ or in block like this $$y = x^2$$
#     
#     
# 
# **Operators:**
# 
# - Add:
#  - $x + y$
# - Subtract:
#  - $x - y$
# - Multiply
#  - $x * y$
#  - $x \times y$ 
#  - $x . y$ 
# - Divide
#  - $x / y$
#  - $x \div y$
#  - $\frac{x}{y}$
# 
# Code:
# 
#     markdown
#     - Add:
#      - $x + y$
#     - Subtract:
#      - $x - y$
#     - Multiply
#      - $x * y$
#      - $x \times y$ 
#      - $x . y$ 
#     - Divide
#      - $x / y$
#      - $x \div y$
#      - $\frac{x}{y}$
#           
# **Relations:**
# 
# - $\pi \approx 3.14159$
# - ${1 \over 0} \neq \inf$
# - $0 < x > 1$
# - $0 \leq x \geq 1$
# 
# Code:
#     
#     markdown
#     $\pi \approx 3.14159$ 
#     ${1 \over 0} \neq \inf$
#     $0 < x > 1$
#     $0 \leq x \geq 1$
#     
# **Fractions:**
# 
# - $^1/_2$
# - $\frac{1}{2x}$
# - ${3 \over 4}$
# 
# 
# Code:
#     
#     markdown
#     $^1/_2$
#     $\frac{1}{2x}$
#     ${3 \over 4}$
# 
# 
# **Greek Alphabet:**
# 
# | Small Letter             | Capical Letter      | Alervative                   |
# | ----------------------- | ------------------- | ---------------------------- |
# | $\alpha$ \alpha         | $A$  A              |                              |
# | $\beta$ \beta           | $B$  B              |                              |
# | $\gamma$ \gamma         | $\Gamma$ \Gamma     |                              |
# | $\delta$ \delta         | $\Delta$ \Delta     |                              |
# | $\epsilon$ \epsilon     | $E$ E               | $\varepsilon$ \varepsilon    |
# | $\zeta$ \zeta           | $Z$ z               |                              |
# | $\eta$ \eta             | $H$ H               |                              |
# | $\theta$ \theta         | $\Theta$ \Theta     | $\vartheta$ \vartheta        |
# | $\iota$ \iota           | $I$ I               |                              |
# | $\kappa$ \kappa         | $K$ K               | $\varkappa$ \varkappa        |
# | $\lambda$ \lambda       | $\Lambda$ \Lambda   |                              |
# | $\mu$ \mu               | $M$ M               |                              |
# | $\nu$ \nu               | $N$ N               |                              |
# | $\xi$ \xi               | $\Xi$ \Xi           |                              |
# | $\omicron$ \omicron     | $O$ O               |                              |
# | $\pi$ \pi               | $\Pi$ \Pi           | $\varpi$ \varpi              |
# | $\rho$ \rho             | $P$ P               | $\varrho$ \varrho            |
# | $\sigma$ \sigma         | $\Sigma$ \Sigma     | $\varsigma$ \varsigma        |
# | $\tau$ \tau             | $T$ T               |                              |
# | $\upsilon$ \upsilon     | $\Upsilon$ \Upsilon |                              |
# | $\phi$ \phi             | $\Phi$ \Phi         | $\varphi$ \varphi            |
# | $\chi$ \chi             | $X$ X               |                              |
# | $\psi$ \psi             | $\Psi$ \Psi         |                              |
# | $\omega$ \omega         | $\Omega$ \Omega     |                                              |
# 
# 
# 
# 
# 
# 
# **Power & Index:**
# 
# You can add power using the carrot **^** symbol. If you have more than one character you have to enclose them in a curly brackets.
# 
# $$f(x) = x^2 - x^{1 \over \pi}$$
# 
# For index you can use the underscore symbol:
# 
# $$f(X,n) = X_n + X_{n-1}$$
# 
# Code:
# 
#     markdown
#     $$f(x) = x^2 - x^{1 \over \pi}$$
#     $$f(X,n) = X_n + X_{n-1}$$
#     
# **Roots & Log:**
# 
# You can express a square root in LaTeX using the **\sqrt** and to change the level of the root you can use **\sqrt[n]** where **n** is the level of the root.
# 
# $$f(x) = \sqrt[3]{2x} + \sqrt{x-2}$$
# 
# To represent a log use **\log[base]** where base is the base of the logarithmic term.
# 
# $$\log[x]x = 1$$
# 
# Code:
# 
#     markdown
#     $$f(x) = \sqrt[3]{2x} + \sqrt{x-2}$$
#     $$\log[x]x = 1$$
#     
# **Sums & Products:**
# 
# You can represent a sum with a sigma using **\sum\limits_{a}^{b}** where a and b are the lower and higher limits of the sum.
# 
# $$\sum\limits_{x=1}^{\infty} {1 \over x} = 2$$
# 
# Also you can represent a product with **\prod\limits_{a}^{a}** where a and b are the lower and higher limits.
# 
# $$\prod\limits_{i=1}^{n} x_i - 1$$
# 
# Code:
#     
#     markdown
#     $$\sum\limits_{x=1}^{\infty} {1 \over x} = 2$$
#     $$\prod\limits_{i=1}^{n} x_i - 1$$
#     
# **Statistics:**
# 
# To represent basic concepts in statistics about sample space S, you can represent a maximum:
# 
# $$max(S) = \max\limits_{i: s_i \in \{S\}} s_i$$
# 
# In the same way you can get the minimum:
# 
# $$min (S) = \min\limits_{i: s_i \in \{S\}} s_i$$
# 
# To represent a binomial coefficient with n choose k, use the following:
# 
# $$\frac{n!}{k!(n-k)!} = {n \choose k}$$
# 
# Code:
#     
#     markdown
#     $$max(S) = \max\limits_{i: s_i \in \{S\}} s_i$$
#     $$min (S) = \min\limits_{i: s_i \in \{S\}} s_i$$
#     $$\frac{n!}{k!(n-k)!} = {n \choose k}$$
#     
# **Calculus:**
# 
# Limits are represented using **\lim\limits_{x \to a}** as x approaches a.
# 
# $$\lim\limits_{x \to 0^+} {1 \over 0} = \inf$$
# 
# For integral equations use **\int\limits_{a}^{b}** where a and b are the lower and higher limits.
# 
# $$\int\limits_a^b 2x \, dx$$
# 
# Code:
# 
#     markdown
#     $$\lim\limits_{x \to 0^+} {1 \over 0} = \inf$$
#     $$\int\limits_a^b 2x \, dx$$
#     
#     
# **Function definition over periods:**
# 
# Defining a function that is calculated differently over a number of period can done using LaTeX. There are a few tricks that we will use to do that:
# 
# - The large curly bracket \left\\{ ... \right. Notice it you want to use ( or \[ you don't have to add a back slash(\\). You can also place a right side matching bracket by replacing the . after \right like this .right}
# - Array to hold the definitions in place. it has two columns with left alignment.  **\begin{array}{ll} ... \end.{array}**
# - Line Breaker **\\\**
# - Text alignment box **\mbox{Text}**
# 
# $f(x) =
# \left\{
#     \begin{array}{ll}
#         0  & \mbox{if } x = 0 \\
#         {1 \over x} & \mbox{if } x \neq 0
#     \end{array}
# \right.$
# 
# Code:
# 
#     markdown
#     $f(x) =
#     \left\{
#         \begin{array}{ll}
#             0  & \mbox{if } x = 0 \\
#             {1 \over x} & \mbox{if } x \neq 0
#         \end{array}
#     \right.$
# 
# Note: If you are planning to show your notebook in NBViewer write your latex code in one line. For example you can write the code above like this:
# 
#     markdown
#     $f(x) =\left\{\begin{array}{ll}0  & \mbox{if } x = 0 \\{1 \over x} & \mbox{if } x \neq 0\end{array}\right.$
