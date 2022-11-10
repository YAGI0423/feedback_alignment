### 이 저장소(Repository)는 「Feedback Aligment를 이용한 신경망 학습 알고리즘 구현」에 대한 내용을 다루고 있습니다.

***
작성자: YAGI<br>

최종 수정일: 2022-11-10
+ 2022.11.10: 코드 작성 완료(Task 1 ~ 3)
***
<br>

***
+ 프로젝트 기간: 2022-06-27 ~ (진행 중)

***
<br>

***
+ 해당 프로젝트는 Timothy P. Lillicrap 외 3인의 「Random feedback weights support learning in deep neural networks」(2014)을 바탕으로 하고 있습니다.

> Timothy P. Lillicrap, Daniel Cownden, Douglas B. Tweed, Colin J. Akerman. Random feedback weights support learning in deep neural networks. [ArXiv, 1411.0247v1, 2014](https://arxiv.org/abs/1411.0247).
***
<br><br>

### 프로젝트 요약
***
&nbsp;&nbsp;
오차 역전파(Backpropagation of error)는 현재 가장 강력한 딥러닝 네트워크 학습 알고리즘이다. 하지만, 역전파는 뉴런이 기여하는 영향을 정확하게 계산하여 오류 신호를 하류의 뉴런에 할당하는데, 이는 생물학적으로 수용하기 어렵다. Timothy P. Lillicrap 외 3인은 역전파에서 사용하는 가중치의 전치 대신, '무작위 시냅스 가중치(random synaptic weights)'를 오류 신호와 곱하여 영향을 할당하는 Feedback Alignment 알고리즘(이하 FA)을 제시하였다. 나아가, 특정 작업에 대한 FA 알고리즘의 성능을 역전파 알고리즘과 비교하여 확인하였다. 성능 비교는 
*Task (1) 선형 함수 근사*, *Task (2) MNIST 데이터셋*, *Task (3) 비선형 함수 근사*를 통해 이루어졌다. **본 프로젝트는 앞선 세 작업의 재현을 목표로 한다.**

<br>

**Task (1) Linear function approximation**

&nbsp;&nbsp; 본문

<img src='./README_Figures/figure1_d.png' width='450'>

**figure 1.** Error on Test Set of Paper's Task (1) Linear function approximation

본문

<br>

<img src='./README_Figures/task1_linearFunction.png' width='900'>

**figure 2.** Error of Project's Task (1) Linear function approximation

본문

<br>

**Task (2) MNIST dataset**

<br>

**Task (3) Nonlinear function approximation**
***

<br><br>

### Getting Started
***
본문
***

<br><br>

### License
***
This project is licensed under the terms of the [MIT license](https://github.com/YAGI0423/feedback_alignment/blob/main/LICENSE).
***