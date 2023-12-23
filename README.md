# final_ML_personal
Nguyễn Thị Thu Hồng - 52100962

# DỰ ÁN CUỐI KỲ
# NHẬP MÔN HỌC MÁY
BÀI 1: Trình bày một bài nghiên cứu, đánh giá của em về các vấn đề sau:
1.	Tìm hiểu, so sánh các phương pháp Optimizer trong huấn luyện mô hình học máy;
2.	Tìm hiểu về Continual Learning và Test Production khi xây dựng một giải pháp học máy để giải quyết một bài toán nào đó;

## CHƯƠNG 1 – TỔNG QUAN VỀ HỌC MÁY
### 1.1 Học máy là gì?
Machine learning (học máy hay máy học) là một nhánh con của trí tuệ nhân tạo (AI) và khoa học máy tính. Machine learning sử dụng dữ liệu, thuật toán đầu vào để tự xử lý các vấn đề và liên tục tối ưu để tạo ra những phương án xử lý mới hiệu quả hơn, phù hợp hơn, giống như cách thức tự học của não bộ con người.

Một cách tổng quát, trong cuốn sách Machine Learning của tác giả Tom Mitchell xuất bản năm 1997, học máy được định nghĩa như sau: “A computer program is said to learn to perform a task T from experience E, if its performance at task T, as measured by a performance metric P, improves with experience E over time” (Một chương trình máy tính được cho là học để thực hiện một nhiệm vụ T từ kinh nghiệm E, nếu hiệu suất thực hiện công việc T của nó được đo bởi chỉ số hiệu suất P và được cải thiện bởi kinh nghiệm E theo thời gian).
Khác biệt giữa chương trình lập trình truyền thống và học máy.

<img src="picture/1.1.png">

Hình 1.1 Minh họa chương trình lập trình truyền thống

<img src="picture/1.1.1.png">

Hình 1.1 Minh họa học máy

Thống kê và dự đoán là hai mục đích chính của việc áp dụng machine learning vì thế hệ thống này được thiết kế với khả năng tự nghiên cứu, cải tiến bản thân dựa trên những nguyên lý được lập trình ban đầu. Trong nhiều trường hợp machine learning sẽ tự đề xuất ra giải pháp tối ưu mà không cần được lập trình trước. Do đó, có thể nói Machine Learning giống như một người lao động với khả năng tự học, hoàn thiện và giàu kinh nghiệm hơn theo thời gian.

Trong những năm gần đây, khi mà khả năng tính toán của các máy tính được nâng lên một tầm cao mới và lượng dữ liệu khổng lồ được thu thập bởi các hãng công nghệ lớn, Machine Learning đã tiến thêm một bước dài và một lĩnh vực mới được ra đời gọi là Deep Learning (Học Sâu). Deep Learning đã giúp máy tính thực thi những việc tưởng chừng như không thể vào 10 năm trước: phân loại cả ngàn vật thể khác nhau trong các bức ảnh, tự tạo chú thích cho ảnh, bắt chước giọng nói và chữ viết của con người, giao tiếp với con người, hay thậm chí cả sáng tác văn hay âm nhạc, …

<img src="picture/1.2.png">

Hình 1.2 Mối quan hệ giữa AI, Machine Learning và Deep Learning

(Nguồn: What’s the Difference Between Artificial Intelligence, Machine Learning, and Deep Learning?)

### 1.2 Phân loại Học máy:

Dựa trên các tiêu chí khác nhau, người ta có thể phân loại các thuật toán Học máy theo nhiều cách khác nhau. 

#### 1.2.1 Phân loại theo vấn đề, nhiệm vụ cần giải quyết:

Dựa vào vấn đề, nhiệm vụ cần giải quyết của thuật toán, người ta phân loại các thuật toán Học máy thành ba loại:

1.	Hồi quy (Regression): Giải quyết bài toán dự đoán giá trị một đại lượng nào đó dựa vào giá trị của các đại lượng liên quan. Ví dụ, dựa vào các đặc điểm như diện tích, số phòng, khoảng cách tới trung tâm…để dự đoán giá trị căn nhà.
2.	Phân lớp (Classification): Giải quyết các bài toán nhận dạng xem một đối tượng thuộc lớp nào trong số các lớp cho trước. Ví dụ, bài toán nhận diện chữ viết, bài toán phân loại email…thuộc các thuật toán phân lớp.
3.	Phân cụm (Clustering): Ý tưởng cơ bản giống với các thuật toán phân lớp, sự khác biệt là ở chỗ, trong các bài toán phân cụm, các cụm chưa được xác định trước và thuật toán phải tự khám phá và phân cụm dữ liệu.

<img src="picture/1.3.png">

Hình 1.3 Các giải thuật Học máy

(Nguồn: https://tailieuhay.vn/tai-lieu/bai-giang-hoc-may-bai-5-cay-phan-loai-va-hoi-quy-nguyen-thanh-tung-7733/ )

#### 1.2.2 Phân loại theo cách máy tính học:

Dựa trên cách máy tính học, người ta chia các thuật toán Học máy thành ba loại:

1.	Học tập dưới sự giám sát (Supervised learning): Con người sẽ lập trình dữ liệu đầu vào bao gồm cả cách thức và phương án mà con người mong muốn. Phương án và đáp án sẽ được gắn nhãn, sắp xếp sẵn và Machine Learning chỉ cần rà soát và trả ra đúng kết quả có trong bộ dữ liệu đã có. Tin nhắn rác đến từ 1 số nguồn sẽ tự động được tách ra khỏi hộp thư chính là ứng dụng của machine learning giúp phân loại tin nhắn trên email.
2.	Học tập mà không giám sát (Unsupervised learning): Machine learning chỉ được cung cấp các thuật toán, công cụ để tự xử lý mà không biết trước kết quả. Dễ thấy nhất việc ứng dụng của phân loại này đó là cá nhân hóa trải nghiệm khách hàng.Dữ liệu đầu vào bao gồm hành vi, lịch sử mua mua hàng và hệ thống sẽ dự đoán những sản phẩm phù hợp và đề xuất riêng cho từng khách hàng.
3.	Học tập được giám sát bán phần (Semi-supervised learning): Đây là phân loại nằm ở giữa của 2 phân loại trên khi này dữ liệu đầu vào là 1 hỗn hợp bao gồm cả phương pháp lẫn đáp án. Điểm khác biệt ở đây là phương án và đáp án chưa được nhóm lại thành từng bộ. Như vậy machine learning phải tự tìm ra cách giải nào tương thích với đáp án nào trong bộ dữ liệu sẵn có.

### 1.3 Các bước cơ bản thực hiện một thuật toán Học máy:

Nhìn chung, việc thực hiện một thuật toán Học máy thường trải qua các bước cơ bản sau:

1.	Thu thập dữ liệu – Gathering data/Data collection
2.	Tiền xử lý dữ liệu – Data preprocessing
    1.	Trích xuất dữ liệu – data extraction
    2.	Làm sạch dữ liệu – data cleaning
    3.	Chuyển đổi dữ liệu – Data transformation
    4.	Chuẩn hóa dữ liệu – Data normalization
    5.	Trích xuất đặc trưng – Feature extraction
3.	Phân tích dữ liệu – Data analysis
4.	Xây dựng mô hình máy học – Model building
5.	Huấn luyện mô hình – Model training
6.	Đánh giá mô hình – Model evaluation
   
Trong tất cả các bước thì việc thu thập dữ liệu, tiền xử lý và xây dựng bộ dữ liệu là tốn nhiều thời gian và công sức nhất. Đây là bước quan trọng, có ảnh hưởng rất nhiều đến hiệu quả của thuật toán Học máy.

### 1.4 Ứng dụng của Học máy:

Ứng dụng tổng quát:

•	Xử lý ảnh
•	Phân tích văn bản
•	Khai phá dữ liệu

Ứng dụng trong thực tế:

•	Giải mã thị trường tài chính

•	Thay đổi cục diện ngành nông nghiệp

•	Nâng cao hiệu quả và cải thiện chất lượng dịch vụ ngành y tế

•	Cơ quan nhà nước có thể quản lý trật tự xã hội và đảm bảo tình hình phát triển đất nước

## CHƯƠNG 2 – CÁC PHƯƠNG PHÁP OPTIMIZER
### 2.1 Tổng quan về Optimizer
#### 2.1.1 Optimizer là gì?
Optimizer hay Thuật toán tối ưu là cơ sở để xây dựng mô hình Neural Network với mục đích “học” được các feature (hay pattern) của dữ liệu đầu vào, để từ đó có thể tìm một tập các trọng số (weights – w) và ngưỡng (bias – b) phù hợp hơn để tối ưu hóa mô hình. 

Và có thể nói các thuật toán tối ưu (Optimizition Algorithm) là một trong những “hạt nhân” mạnh mẽ của hầu hết thuật toán Machine Learning. Đây một quy trình được thực hiện lặp đi lặp lại bằng cách so sánh các giải pháp khác nhau cho đến khi tìm thấy một giải pháp tối ưu hoặc thỏa đáng.

Đối với kỹ thuật học sâu nói riêng, thuật toán tối ưu là các kỹ thuật giúp xây dựng các mô hình mạng nơ-ron để tối ưu hóa độ chính xác của mô hình mạng.

<img src="picture/2.1.png">

Hình 2.1 Minh họa thuật toán tối ưu (Optimizer)

#### 2.1.2 Vai trò của thuật toán tối ưu

Trong thuật toán học máy nói chung và kỹ thuật học sâu nói riêng, thuật toán tối ưu hóa (Optimizer) là một khâu quan trọng không thể thiếu. Quá trình tối ưu hóa thực hiện xác định hàm mất mát (loss function) và sau đó tối thiểu hóa hàm trên bằng cách sử dụng hàm tối ưu. Cụ thể, thông qua việc cập nhật các tham số của mô hình (w, b) và đánh giá lại hàm mất mát với một tỉ lệ học (learning rate) xác định, quá trình tối ưu giúp mô hình tương thích tốt hơn với tập dữ liệu được đào tạo.

##### 2.1.2.1 Hàm mất mát (Loss function)

Hàm mất mát (Loss function) là một phương pháp đánh giá độ hiệu quả của thuật toán “học” cho mô hình trên tập dữ liệu được sử dụng. 
Hàm mất mát trả về một số thực không âm thể hiện sự chênh lệch giữa hai đại lượng: 

•	a: nhãn được dự đoán

•	y: nhãn đúng

Bản thân hàm mất mát chính là một cơ chế thưởng-phạt, mô hình sẽ phải đóng phạt mỗi lần dự đoán sai và mức phạt tỉ lệ thuận với độ lớn sai sót. 
Trong mọi bài toán học có giám sát, mục tiêu luôn bao gồm giảm tổng mức phạt phải đóng. Trong trường hợp lý tưởng a = y, hàm mất mát sẽ trả về giá trị cực tiểu bằng 0. 
Hai hàm mất mát thường xuyên được sử dụng trong mạng nơ-ron: 

•	MSE (Mean Squared Error)

•	Cross Entropy

##### 2.1.2.2 Tỉ lệ học (Learning rate)

Learning rate hay tỉ lệ học là một thông số quan trọng trong việc quyết định tốc độ học của mạng nơ-ron. Tốc độ học được thể hiện bằng sự thay đổi giá trị cập nhật trọng số (weights - w) trong các chu kỳ học. Tùy theo mục đích của mô hình mà tăng/ giảm tỉ lệ học. 

Tỉ lệ học càng cao thì giúp mô hình học khá nhanh và tiết kiệm được thời gian huấn luyện, tuy nhiên việc tỉ lệ học lớn đồng nghĩa với việc sự thay đổi trọng số (weights - w) và tham số ngưỡng (bias - b) càng lớn, mô hình không ổn định, một số chu kỳ học có sự dao động mạnh ở tỉ lệ nhận dạng đúng hay nói cách khác là thuật toán không được tối ưu và ngược lại đối với tỉ lệ học nhỏ.

#### 2.1.3 Yếu tố đánh giá một thuật toán tối ưu

Một vài các yếu tố hay được sử dụng để đánh giá một thuật toán tối ưu (Optimizer):

•	Hội tụ nhanh (trong quá trình train)

•	Sự tổng quát hóa cao (vẫn nhận dạng được những mẫu chưa từng được huấn luyện)

•	Độ chính xác cao

### 2.2 Một số thuật toán tối ưu (Optimization Algorithms)

Một số thuật toán tối ưu phổ biến:

1.	Gradient Descent
2.	SGD với động lượng
3.	RMSProp
4.	Adagrad
5.	Adadelta
6.	Adam
7.	AdamW
8.	AMSGrad
   
#### 2.2.1 Gradient Descent (GD)

Gradient Descent (GD) là thuật toán tìm tối ưu chung cho các hàm số. Ý tưởng chung của GD là điều chỉnh các tham số để lặp đi lặp lại thông qua mỗi dữ liệu huấn luyện để giảm thiểu hàm chi phí. 

𝑤(𝑘+1) = 𝑤(𝑘) − 𝜂 ∇𝑤 𝐽(𝑤(𝑘) )

Trong đó:

•	𝑤(𝑘) : tham số tại bước cập nhật tại lớp k

•	Η: tỉ lệ học

•	𝐽(𝑤): hàm lỗi

•	∇𝑤 𝐽(𝑤 (𝑘)): đạo hàm của hàm lỗi tại điểm 𝑤(𝑘)

Ví dụ:

```sh
from __future__ import division, print_function, unicode_literals
import math
import numpy as np 
import matplotlib.pyplot as plt

def grad(x):
    return 2*x+ 5*np.cos(x)

def cost(x):
    return x**2 + 5*np.sin(x)

def myGD1(eta, x0):
    x = [x0]
    for it in range(100):
        x_new = x[-1] - eta*grad(x[-1])
        if abs(grad(x_new)) < 1e-3:
            break
        x.append(x_new)
    return (x, it)

(x1, it1) = myGD1(.1, -5)
(x2, it2) = myGD1(.1, 5)
print('Solution x1 = %f, cost = %f, obtained after %d iterations'%(x1[-1], cost(x1[-1]), it1))
print('Solution x2 = %f, cost = %f, obtained after %d iterations'%(x2[-1], cost(x2[-1]), it2))
```
Kết quả:

```sh
Solution x1 = -1.110667, cost = -3.246394, obtained after 11 iterations
Solution x2 = -1.110341, cost = -3.246394, obtained after 29 iterations
```
	Điểm khởi tạo khác nhau

Sau khi có các hàm cần thiết, tôi thử tìm nghiệm với các điểm khởi tạo khác nhau là x0 = −5 và x0 = 5.

<img src="picture/2.2.png"> <img src="picture/2.2.1.png">

Hình 2.2 Minh họa thuật toán GD với điểm khởi tạo khác nhau

Từ hình minh họa trên ta thấy rằng ở hình bên trái, tương ứng với x0 =−5, nghiệm hội tụ nhanh hơn, vì điểm ban đầu x0 gần với nghiệm x* ≈ −1 hơn. Hơn nữa, với x0 =5 ở hình bên phải, đường đi của nghiệm có chứa một khu vực có đạo hàm khá nhỏ gần điểm có hoành độ bằng 2. 

=>	Điều này khiến cho thuật toán la cà ở đây khá lâu. Khi vượt qua được điểm này thì mọi việc diễn ra rất tốt đẹp.

	Learning rate khác nhau

Tốc độ hội tụ của GD không những phụ thuộc vào điểm khởi tạo ban đầu mà còn phụ thuộc vào learning rate. 

Ví dụ với cùng điểm khởi tạo x0 = −5 nhưng learning rate khác nhau:

<img src="picture/2.3.png"> <img src="picture/2.3.1.png">

Hình 2.3 Minh họa thuật toán GD với Learning rate khác nhau

Ta quan sát thấy hai điều:

1.	Với learning rate nhỏ η=0.01, tốc độ hội tụ rất chậm. Trong ví dụ, do chọn tối đa 100 vòng lặp nên thuật toán dừng lại trước khi tới đích, mặc dù đã rất gần. Trong thực tế, khi việc tính toán trở nên phức tạp, learning rate quá thấp sẽ ảnh hưởng tới tốc độ của thuật toán rất nhiều, thậm chí không bao giờ tới được đích.
2.	Với learning rate lớn η=0.5, thuật toán tiến rất nhanh tới gần đích sau vài vòng lặp. Tuy nhiên, thuật toán không hội tụ được vì bước nhảy quá lớn, khiến nó cứ quẩn quanh ở đích.
   
	Việc lựa chọn learning rate rất quan trọng trong các bài toán thực tế. Việc lựa chọn giá trị này phụ thuộc nhiều vào từng bài toán và phải làm một vài thí nghiệm để chọn ra giá trị tốt nhất. Ngoài ra, tùy vào một số bài toán, GD có thể làm việc hiệu quả hơn bằng cách chọn ra learning rate phù hợp hoặc chọn learning rate khác nhau ở mỗi vòng lặp.

Có một số biến thể khác nhau của GD tùy thuộc vào số lượng dữ liệu được sử dụng để tính gradient của hàm mất mát. Gồm: 

1.	Batch Gradient Descent (Batch GD)
2.	Stochastic Gradient Descent (SGD)
3.	Mini-batch Gradient Descent (Mini-batch GD)
   
##### 2.2.1.1 Batch Gradient Descent (Batch GD)

Thuật toán Batch Gradient Descent (Batch GD) tính gradient của hàm mất mát tại w trên toàn bộ tập dữ liệu. Tất cả các điểm dữ liệu đều được sử dụng để tính gradient trước khi cập nhật bộ trọng số w. Hạn chế của Batch GD là khi tập dữ liệu lớn, việc tính gradient sẽ tốn nhiều thời gian và chi phí tính toán.

##### 2.2.1.2 Stochastic Gradient Descent (SGD)

Để khắc phục hạn chế của Bathc GD, thuật toán Stochastic Gradient Descent (SGD) thực hiện việc cập nhật trọng số với mỗi mẫu dữ liệu x(i) có nhãn tương ứng y(i) như sau: 

𝑤(𝑘+1) = 𝑤(𝑘) − 𝜂 ∇𝑤 𝐽(𝑤(𝑘), x(i), y(i))

Với cách cập nhật này, SGD thường nhanh hơn Batch GD và có thể sử dụng để học trực tuyến (online learning) khi tập dữ liệu huấn luyện được cập nhật liên tục. 

Với SGD, bộ trọng số w được cập nhật thường xuyên hơn so với Batch GD và vì vậy hàm mất mát cũng dao động nhiều hơn. Sự dao động này khiến SGD có vẻ không ổn định nhưng lại có điểm tích cực là nó giúp di chuyển đến những điểm cực tiểu (địa phương) mới có tiềm năng hơn. Với tốc độ học giảm, khả năng hội tụ của SGD cũng tương đương với Batch GD.

Hàm số trong Python để giải Linear Regression theo SGD:

```sh
def sgrad(w, i, rd_id):
    true_i = rd_id[i]
    xi = Xbar[true_i, :]
    yi = y[true_i]
    a = np.dot(xi, w) - yi
    return (xi*a).reshape(2, 1)

def SGD(w_init, grad, eta):
    w = [w_init]
    w_last_check = w_init
    iter_check_w = 10
    N = X.shape[0]
    count = 0
    for it in range(10):  
# shuffle data 
        rd_id = np.random.permutation(N)
        for i in range(N):
            count += 1 
            g = sgrad(w[-1], i, rd_id)
            w_new = w[-1] - eta*g
            w.append(w_new)
            if count%iter_check_w == 0:
                w_this_check = w_new                 
                if np.linalg.norm(w_this_check - w_last_check)/len(w_init) < 1e-3:                                    
                    return w
                w_last_check = w_this_check
    return w
```

Kết quả thu được:

<img src="picture/2.4.png"> <img src="picture/2.4.1.png">
Hình 2.4 Trái: đường đi của nghiệm với SGD. Phải: giá trị của Loss function tại 50 vòng lặp đầu tiên.

##### 2.2.1.3 Mini-batch Gradient Descent (Mini-batch GD)

Cách tiếp cận thứ ba là thuật toán Mini-batch Gradient Descent (Mini-batch GD). Khác với hai thuật toán trước, Mini-batch GD sử dụng t điểm dữ liệu để cập nhật bộ trọng số (1<t<N) với N là tổng số điểm dữ liệu). 

𝑤(𝑘+1) = 𝑤(𝑘) − 𝜂 ∇𝑤 𝐽(𝑤(𝑘), x(i: i+t), y(i: i+t))

Với x(i: i+t) được hiểu là dữ liệu từ thứ i tới thứ i+t−1. Dữ liệu này sau mỗi epoch là khác nhau vì chúng cần được xáo trộn. Một lần nữa, các thuật toán khác cho GD như Momentum, Adagrad, Adadelta, … cũng có thể được áp dụng vào đây. Giá trị t thường được chọn là khoảng từ 50 đến 100. 

Mini-batch GD giảm sự dao động của hàm mất mát so với SGD và chi phí tính gradient với k điểm dữ liệu là chấp nhận được. 

Mini-batch GD thường được lựa chọn khi huấn luyện mạng nơron và vì vậy trong một số trường hợp, SGD được hiểu là Mini-batch GD. Riêng bản thân Mini-batch GD không đảm bảo tìm được điểm cực tiểu của hàm mất mát mà bên cạnh đó các yếu tố như tốc độ học, thuộc tính dữ liệu và tính chất của hàm mất mát cũng ảnh hưởng đến điều này.

Ví dụ về giá trị của hàm mất mát mỗi khi cập nhật tham số w của một bài toán khác:

<img src="picture/2.5.png">
Hình 2.5 Ví dụ về Mini-batch Gradient Descent

	Hàm mất mát nhảy lên nhảy xuống (fluctuate) sau mỗi lần cập nhật nhưng nhìn chung giảm dần và có xu hướng hội tụ về cuối.

#### 2.2.2 SGD với động lượng (SGD with momentum)

SGD với momentum là phương pháp giúp tăng tốc các vectơ độ dốc theo đúng hướng, và giúp hệ thống hội tụ nhanh hơn. Đây là một trong những thuật toán tối ưu hóa phổ biến nhất và nhiều mô hình hiện đại sử dụng nó để đào tạo. 

Mô tả như sau: 

𝑣𝑗 ← 𝛼 ∗ 𝑣𝑗 − 𝜂 ∗ 𝛻𝑊 ∑_1^m▒〖L_m (w) 〗 

𝑤𝑗 ← 𝑣𝑗 + 𝑤𝑗

Phương trình có hai phần. Trong đó:

	vj: độ dốc được giữ lại từ các lần lặp trước
 
	Hệ số động lượng α: tỉ lệ phần trăm của độ dốc được giữ lại mỗi lần lặp
 
	L: hàm mất mát
 
	η: tỉ lệ học

#### 2.2.3 RMSProp (Root Mean Square Propogation)

RMSProp sử dụng trung bình bình phương của gradient để chuẩn hóa gradient. Có tác dụng cân bằng kích thước bước - giảm bước cho độ dốc lớn để tránh hiện tượng phát nổ độ dốc (Exploding Gradient), và tăng bước cho độ dốc nhỏ để tránh biến mất độ dốc (Vanishing Gradient). RMSProp tự động điều chỉnh tốc độ học tập, và chọn một tỉ lệ học tập khác nhau cho mỗi tham số. 

Phương pháp cập nhật các trọng số được thực hiện như mô tả:

𝑠𝑡 = 𝜌𝑠𝑡−1 + (1 − 𝜌) ∗ g_t^2

𝛥𝑥𝑡 = -η/√(s_t  + ϵ) ∗ 𝑔𝑡 

𝑥𝑡+1 = 𝑥𝑡 + 𝛥𝑥t

Trong đó:

	𝑠𝑡: tích luỹ phương sai của các gradient trong quá khứ
 
	𝜌: tham số suy giảm
 
	𝛥𝑥𝑡: sự thay đổi các tham số trong mô hình
 
	𝑔𝑡: gradient của các tham số tại vòng lặp t
 
	ϵ: tham số đảm bảo kết quả xấp xỉ có ý nghĩa.
 
#### 2.2.4 Adagrad

Adagrad là một kỹ thuật học máy tiên tiến, thực hiện giảm dần độ dốc bằng cách thay đổi tốc độ học tập. Adagrad được cải thiện hơn bằng cách cho trọng số học tập chính xác dựa vào đầu vào trước nó để tự điều chỉnh tỉ lệ học theo hướng tối ưu nhất thay vì với một tỉ lệ học duy nhất cho tất cả các nút.

Thuật toán Adagrad được Duchi J. và các cộng sự đề xuất năm 2011. Khác với SGD, tốc độ học trong Adagrad thay đổi tùy thuộc vào trọng số: tốc độ học thấp đối với các trọng số tương ứng với các đặc trưng phổ biến, tốc độ học cao đối với các trọng số tương ứng với các đặc trưng ít phổ biến.

gt, i  = ∇wJ (wt, i)

Trong đó:

	gt: gradient của hàm mất mát tại bước t
 
	gt, i :  đạo hàm riêng của hàm mất mát theo wi tại bước t
 
Quy tắc cập nhật của Adagrad:

w𝑡+1, i = wt, i -η/√(G_(t,ii)  + ϵ) ∗ 𝑔𝑡, i

Theo quy tắc cập nhật, Adagrad điều chỉnh tốc độ học η tại bước t tương ứng với trọng số wi xác định dựa trên các gradient đã tính được theo wi. 

	Mẫu số là chuẩn L2 (L2 norm) của ma trận đường chéo Gt trong đó phần tử i,i là tổng bình phương của các gradient tương ứng với wi tính đến bước t.
 
	ε là một số dương khá nhỏ nhằm tránh trường hợp mẫu số bằng 0.
 
Quy tắc cập nhật trên có thể viết dưới dạng tổng quát hơn như sau:

w𝑡+1 = wt -η/√(G_t  + ϵ) ⨀𝑔𝑡

Trong đó, ⨀ là phép nhân ma trận-vectơ giữa Gt và gt . 

Có thể nhận thấy rằng trong thuật toán Adagrad tốc độ học được tự động điều chỉnh. Adagrad thường khá hiệu quả đối với bài toán có dữ liệu phân mảnh. Tuy nhiên, hạn chế của Adagrad là các tổng bình phương ở mẫu số ngày càng lớn khiến tốc độ học ngày càng giảm và có thể tiệm cận đến giá trị 0 khiến cho quá trình huấn luyện gần như đóng băng. Bên cạnh đó, giá trị tốc độ học η cũng phải được xác định một cách thủ công.

#### 2.2.5 Adadelta

Thuật toán Adadelta được Zeiler và các cộng sự đề xuất năm 2012. Adadelta là một biến thể của Adagrad để khắc phục tình trạng giảm tốc độ học ở Adagrad. Adadelta không có tham số tỉ lệ học cho nên, thay vì lưu lại tất cả gradient như Adagrad, Adadelta giới hạn tích lũy gradient theo cửa sổ có kích thước cố định của trọng số w. Bằng cách này, Adadelta vẫn tiếp tục học sau nhiều bước cập nhật.

g_t^'= √((〖Δx〗_(t-1)+ϵ)/(s_t+ϵ)) *𝑔𝑡 

𝑥𝑡 = 𝑥𝑡−1 − g_t^' 

𝛥𝑥𝑡 = 𝜌𝛥𝑥𝑡−1 + (1 − 𝜌) x_t^2

Từ công thức, Adadelta sử dụng 2 biến trạng thái: 

	𝑠𝑡: để lưu trữ trung bình của khoảng thời gian thứ hai của gradient và Δ𝑥𝑡 để lưu trữ trung bình của khoảng thời gian thứ 2 của sự thay đổi các tham số trong mô hình. 
 
	g_t^': căn bậc hai thương của trung bình tốc độ thay đổi bình phương và trung bình mô-men bậc hai của gradient. 
 
#### 2.2.6 Adam

Adam được xem như là sự kết hợp của RMSprop và Stochastic Gradient Descent với động lượng. Adam là một phương pháp tỉ lệ học thích ứng, nó tính toán tỉ lệ học tập cá nhân cho các tham số khác nhau. Adam sử dụng ước tính của khoảng thời gian thứ nhất và thứ hai của độ dốc để điều chỉnh tỉ lệ học cho từng trọng số của mạng nơ-ron.

Tuy nhiên, qua nghiên cứu thực nghiệm, trong một số trường hợp, Adam vẫn còn gặp phải nhiều thiếu sót so với thuật toán SGD. Thuật toán Adam được mô tả: 

𝑚𝑡 = 𝛽1𝑚𝑡−1 + (1 − 𝛽1)𝑔𝑡

𝑣𝑡 = 𝛽2𝑣𝑡−1 + (1 − 𝛽2) g_t^2

Trong đó:

	vt là trung bình động của bình phương
 
	mt là trung bình động của gradient
 
	β1và β2 là tốc độ của di chuyển
 
#### 2.2.7 AdamW

AdamW là một biến thể của Adam. Ý tưởng của AdamW khá đơn giản: khi thực hiện thuật toán Adam với L2 regularization (chuẩn hóa L2), tác giả loại bỏ phần tiêu biến của trọng số (weight decay) wtθt khỏi công thức tính gradient hàm mất mát tại thời điểm t: 

gt = 𝛥f(θt) + wtθt

và thay vào đó, đưa phần giá trị đã được phân tách này vào quá trình cập nhật trọng số: 

θ_(t+1,i)=θ_(t,i)-η(1/√((v_t ) ̂+ϵ)*(m_t ) ̂+w_(t,i) θ_(t,i) ),∀t

#### 2.2.8 AMSGrad

AMSGrad sử dụng giá trị lớn nhất của các bình phương gradient trước đó vt để cập nhật các trọng số. Ở đây, vt cũng được định nghĩa như trong thuật toán Adam: 

𝑣𝑡 = 𝛽2𝑣𝑡−1 + (1 − 𝛽2) g_t^2

Thay vì trực tiếp sử dụng vt (hay giá trị ước lượng (v_t ) ̂ ), thuật toán sẽ sử dụng giá trị trước đó vt-1 nếu giá trị này lớn hơn giá trị hiện tại: 

(v_t ) ̂=max⁡((v_(t-1) ) ̂,v_t) 

Tương tự như trong thuật toán Adam, các giá trị được ước lượng theo công thức dưới đây để khử lệch cho các trọng số:

𝑚𝑡 = 𝛽1𝑚𝑡−1 + (1 − 𝛽1)𝑔𝑡

𝑣𝑡 = 𝛽2𝑣𝑡−1 + (1 − 𝛽2) g_t^2

(v_t ) ̂=max⁡((v_(t-1) ) ̂,v_t) 

AMSGrad được cập nhật theo quy tắc:

w𝑡+1 = wt -η/√((v_t ) ̂  + ϵ)*m𝑡






