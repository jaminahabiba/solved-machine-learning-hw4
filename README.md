Download Link: https://assignmentchef.com/product/solved-machine-learning-hw4
<br>
<h1></h1>

<ol>

 <li><strong> Max of Convex Functions. </strong>Consider <em>m </em>convex functions <em>f</em><sub>1</sub>(<em>x</em>)<em>,…,f<sub>m</sub></em>(<em>x</em>), where <em>f<sub>i </sub></em>: R<em><sup>d </sup></em>→ R. Now define a new function <em>g</em>(<em>x</em>) = max<em><sub>i </sub>f<sub>i</sub></em>(<em>x</em>).

  <ul>

   <li>Prove that <em>g</em>(<em>x</em>) is a convex function.</li>

   <li>Show that a sub-gradient of <em>g </em>at point <em>x </em>is the gradient of a function <em>f<sub>i </sub></em>(assume <em>f<sub>i </sub></em>is differentiable) for which <em>f<sub>i</sub></em>(<em>x</em>) = max{<em>f</em><sub>1</sub>(<em>x</em>)<em>,…,f<sub>m</sub></em>(<em>x</em>)}.</li>

  </ul></li>

 <li><strong> </strong><em>`</em><sup>2 </sup> Consider the following problem:</li>

</ol>

s.t. <em>y<sub>i</sub></em>(<em>w<sup>T </sup>x<sub>i </sub></em>+ <em>b</em>) ≥ 1 − <em>ξ<sub>i                   </sub></em>∀<em>i </em>= 1<em>,…,m</em>

<ul>

 <li>Show that a constraint of the form <em>ξ<sub>i </sub></em>≥ 0 will not change the problem. meaning, Show that these non-negativity constraints can be removed. That is, show that the optimal value of the objective will be the same whether or not these constraints are present.</li>

 <li>What is the Lagrangian of this problem?</li>

 <li>Minimize the Lagrangian with respect to <em>w,b,ξ </em>by setting the derivative with respect to these variables to 0.</li>

 <li>What is the dual problem?</li>

</ul>

<ol start="3">

 <li><strong> The Multi-class Hinge-loss. </strong>Consider the problem of multi-class prediction where the label <em>Y </em>has <em>L </em>values (e.g., <em>L </em>= 10 for MNIST). Denote [<em>L</em>] = 1<em>,</em>2<em>,…,L</em>. Assume the inputs are <em>x </em>∈ R<em><sup>d</sup></em>. We will consider classifiers of the form <em>f</em>(<em>x</em>;<em>w</em><sub>1</sub><em>,…,w<sub>L</sub></em>) = argmax<em><sub>y</sub></em><sub>∈[<em>L</em>] </sub><em>w<sub>y </sub></em> <em>x </em>defined by <em>L </em>vectors <em>w</em><sub>1</sub><em>,…,w<sub>L</sub></em>. Given an input <em>x </em>and its correct label <em>y </em>the error of the classifier is ∆<em><sub>zo</sub></em>(<em>f</em>(<em>x</em>;<em>w</em><sub>1</sub><em>,…,w<sub>L</sub></em>)<em>,y</em>) where</li>

</ol>

Since this loss is hard to optimize, we consider another loss, called the multi-class hinge loss, and is defined as follows:

<em>.</em>

Given a labeled training set <em>x</em><sub>1</sub><em>,…,x<sub>n </sub></em>∈ R<em><sup>d </sup></em>and <em>y</em><sub>1</sub><em>,…,y<sub>n </sub></em>∈ [<em>L</em>] the hinge-loss optimization problem would be:

<em>.</em>

Denote a minimizer of this problem by (note there may be multiple such minimizers).

1

2                                                                                                                                 <em>Handout Homework 4: April 19, 2020</em>

<ul>

 <li>Show that <em>` </em>is a convex function of <em>w</em><sub>1</sub><em>,…,w<sub>L</sub></em>.</li>

 <li>Show that <em>`</em>(<em>w</em><sub>1</sub><em>,…,w<sub>L</sub>,x,y</em>) ≥ ∆<em><sub>zo</sub></em>(<em>f</em>(<em>x</em>;<em>w</em><sub>1</sub><em>,…,w<sub>L</sub></em>)<em>,y</em>) for all values of <em>w,x,y</em>.</li>

 <li>Assume that for your training set there existsthat achieve zero training error</li>

</ul>

(namely ∆ ) = 0 for all <em>i</em>). Prove that <em>w<sup>opt </sup></em>would also have zero training error. Namely that ∆ ) = 0 for all <em>i</em>.

<ol start="4">

 <li><strong> Growth Function of Composition. </strong>Let and be two function families. Define F = F<sub>2 </sub>◦F<sub>1 </sub>to be the set of functions which are a composition of a function from F<sub>1 </sub>and from F<sub>2</sub>. That is,</li>

</ol>

F = F<sub>2 </sub>◦ F<sub>1 </sub>= {<em>f</em><sub>2 </sub>◦ <em>f</em><sub>1</sub>|<em>f</em><sub>1 </sub>∈ F<sub>1</sub><em>,f</em><sub>2 </sub>∈ F<sub>2</sub>}

Prove that

Π<sub>F</sub>(<em>m</em>) ≤ Π<sub>F</sub><sub>1</sub>(<em>m</em>) · Π<sub>F</sub><sub>2</sub>(<em>m</em>)<em>.</em>

<ol start="5">

 <li><strong>(10 points) Gradient Descent on Smooth Function. </strong>We say that a continuously differentiable function <em>f </em>: R<em><sup>n </sup></em>→ R is <em>β</em>-smooth if for all <strong>x</strong><em>,</em><strong>y </strong>∈ R<em><sup>n</sup></em></li>

</ol>

In words, <em>β</em>-smoothness of a function <em>f </em>means that at every point <strong>x</strong>, <em>f </em>is upper bounded by a qaudratic function which coincides with <em>f </em>at <strong>x</strong>.

Let <em>` </em>: R<em><sup>n </sup></em>→ R be a <em>β</em>-smooth and non-negative function (i.e., <em>`</em>(<strong>x</strong>) ≥ 0 for all <strong>x </strong>∈ R<em><sup>n</sup></em>). Consider the (non-stochastic) gradient descent algorithm applied on <em>` </em>with constant step size <em>η &gt; </em>0: <strong>x</strong><em><sub>t</sub></em><sub>+1 </sub>= <strong>x</strong><em><sub>t </sub></em>− <em>η</em>∇<em>`</em>(<strong>x</strong><em><sub>t</sub></em>)

Assume that gradient descent is initialized at some point <strong>x</strong><sub>0</sub>. Show that if <em>η &lt; <sub>β</sub></em><u><sup>2 </sup></u>then

lim k∇<em>`</em>(<strong>x</strong><em><sub>t</sub></em>)k = 0 <em>t</em>→∞

(Hint: Use the smoothness definition with points <strong>x</strong><em><sub>t</sub></em><sub>+1 </sub>and <strong>x</strong><em><sub>t </sub></em>to show that

∞ and recall that for a sequence implies lim<em><sub>n</sub></em><sub>→∞ </sub><em>a<sub>n </sub></em>= 0. Note that <em>f </em>is not assumed to be convex!)

<em>Handout Homework 4: April 19, 2020                                                                                                                                </em>3

<h1>Programming Assignment</h1>

Submission guidelines:

<ul>

 <li>Download the file skeleton sgd.py from Moodle. In each of the following questions you should only implement the algorithm at each of the skeleton files. Plots, tables and any other artifact should be submitted with the theoretical section.</li>

 <li>In the file skeleton sgd.py there is an helper function. The function reads the examples labelled 0, 8 and returns them with 0-1 labels. Case you are unable to read the MNIST data with the provided script, you can download the file from here:</li>

</ul>

https://github.com/amplab/datasciencesp14/blob/master/lab7/mldata/mnist-original.mat.

<ul>

 <li>Your code should be written in Python 3.</li>

 <li>Make sure to comment out or remove any code which halts code execution, such as matplotlib popup windows.</li>

 <li>Your code submission should include one file: py.</li>

</ul>

<ol>

 <li><strong>(25 points) SGD for Hinge loss. </strong>We will continue working with the MNIST data set. The file template (skeleton sgd.py), contains the code to load the training, validation and test sets for the digits 0 and 8 from the MNIST data. In this exercise we will optimize the Hinge loss (as you seen in the lecture) using the stochastic gradient descent implementation discussed in class. Namely, at each iteration <em>t </em>= 1<em>,… </em>we sample <em>i </em>uniformly; and if <em>y<sub>i</sub>w<sub>t </sub></em> <em>x<sub>i </sub>&lt; </em>1, we update:</li>

</ol>

<em>w</em><em>t</em>+1 = (1 − <em>η</em><em>t</em>)<em>w</em><em>t </em>+ <em>η</em><em>tCy</em><em>ix</em><em>i</em>

and <em>w<sub>t</sub></em><sub>+1 </sub>= (1 − <em>η<sub>t</sub></em>)<em>w<sub>t </sub></em>otherwise, where <em>η<sub>t </sub></em>= <em>η</em><sub>0</sub><em>/t</em>, and <em>η</em><sub>0 </sub>is a constant. Implement an SGD function that accepts the samples and their labels, <em>C</em>, <em>η</em><sub>0 </sub>and <em>T</em>, and runs <em>T </em>gradient updates as specified above. In the questions that follow, make sure your graphs are meaningful.

Consider using set xlim or set ylim to concentrate only on a relevant range of values.

<ul>

 <li><strong>(10 points) </strong>Train the classifier on the training set. Use cross-validation on the validation set to find the best <em>η</em><sub>0</sub>, assuming <em>T </em>= 1000 and <em>C </em>= 1. For each possible <em>η</em><sub>0 </sub>(for example, you can search on the log scale <em>η</em><sub>0 </sub>= 10<sup>−5</sup><em>,</em>10<sup>−4</sup><em>,…,</em>10<sup>4</sup><em>,</em>10<sup>5 </sup>and increase resolution if needed), assess the performance of <em>η</em><sub>0 </sub>by averaging the accuracy on the validation set across 10 runs. Plot the average accuracy on the validation set, as a function of <em>η</em><sub>0</sub>.</li>

 <li><strong>(5 points) </strong>Now, cross-validate on the validation set to find the best <em>C </em>given the best <em>η</em><sub>0 </sub>you found above. For each possible <em>C </em>(again, you can search on the log scale as in section (a)), average the accuracy on the validation set across 10 runs. Plot the average accuracy on the validation set, as a function of <em>C</em>.</li>

 <li><strong>(5 points) </strong>Using the best <em>C</em>, <em>η</em><sub>0 </sub>you found, train the classifier, but for <em>T </em>= 20000. Show the resulting <em>w </em>as an image.</li>

 <li><strong>(5 points) </strong>What is the accuracy of the best classifier on the test set?</li>

</ul>

<ol start="2">

 <li><strong>(15 points) SGD for multi-class cross-entropy. </strong>The skeleton file contains a second helper function to load the training, validation and test sets for all the digits. In this exercise</li>

</ol>

4                                                                                                                                 <em>Handout Homework 4: April 19, 2020</em>

we will optimize the multi-class cross entropy loss using SGD. Recall the multi-class crossentropy loss discussed in the recitation (our classes are 0<em>,</em>1<em>,…,</em>9):

Derive the gradient update for this case, and implement the appropriate SGD function.

<ul>

 <li><strong>(9 points) </strong>Train the classifier on the training set. Use cross-validation on the validation set to find the best <em>η</em><sub>0</sub>, assuming <em>T </em>= 1000. For each possible <em>η</em><sub>0 </sub>(for example, you can search on the log scale <em>η</em><sub>0 </sub>= 10<sup>−5</sup><em>,</em>10<sup>−4</sup><em>,…,</em>10<sup>4</sup><em>,</em>10<sup>5 </sup>and increase resolution if needed), assess the performance of <em>η</em><sub>0 </sub>by averaging the accuracy on the validation set across 10 runs. Plot the average accuracy on the validation set, as a function of <em>η</em><sub>0</sub>.</li>

 <li><strong>(3 points) </strong>Using the best <em>η</em><sub>0 </sub>you found, train the classifier, but for <em>T </em>= 20000. Show the resulting <em>w</em><sub>0</sub><em>,…,w</em><sub>9 </sub>as images.</li>

 <li><strong>(3 points) </strong>What is the accuracy of the best classifier on the test set?</li>

</ul>