# RNNChatBot
This is a simple RNN model implemented by *<font color="red">Python</font>* but has not been optimized.So it was very slow and moreover basic RNN is not practical in practice.But since this is my first implementation of RNN so I think it is deserve to write a readme file to illustrate how RNN works.  

#### Forward Propagation
The basic RNN is very simple.It uses such formula to propagate forward.  
$$
\begin{align}  s_t &= tanh(Ux_t + Ws_{t-1})  \\
o_t &= softmax(Vs_t)  \\
\end{align}
$$

#### Back Propagation Through Time(BPTT)
Differ from normal neural network.RNN used a special back propagation method which is called *<font color="red">Back Propagation Through Time</font>* as also know as *<font color="red">BPTT</font>*.Only looked at this name you almost would know how this algorithm works.Rather than propagate errors between multiple layers of neuron BPTT propagate errors through time.  
The *<font color="red">loss function</font>*  of RNN which is called *<font color="red">Cross Entropy Loss</font>* and it is look like this:
$$\begin{align}
	E_t(y_t,\tilde{y_t}) &= -y_t \log \tilde{y_t} \\
	E(y,\tilde{y_t}) &= \sum_{t} E_t(y_t,\tilde{y_t}) \\
	&= - \sum_{t} y_t \log \tilde{y_t}	
\end{align}$$
In this place you should beware that $y_t$ is the right result at time $t$ of our model.In the beginning I was confused about this terminology and I was also wondering why used *<font color="red">Cross Entropy Loss</font>* here.Then have read some material and I finally figured out why would this happened.In fact you can think cross entropy as an measurement of how two distributions look like.If the differences between two distribution is small their cross entropy is also small.The goal of this model is to find a minimum value of loss function and also give a precise prediction of the next state.So now you can know cross entropy is good loss function in this question.
