#Project
<p>This folder contains sevreal experiments performed over the CNN. In our case, as we mentioned beofre, there are 10 different classes to classify:
<table style="width:100%">
  <tr>
    <td>airport_terminal</td>
    <td>lighthouse</td>
    <td>cemetery</td>
    <td>swimming_pool</td>
    <td>windmill</td>
  </tr>
    <tr>
    <td>bookstore</td>
    <td>castle</td>
    <td>gas_station</td>
    <td>highway</td>
    <td>skyscraper</td>
  </tr>
</table>
All of them are originally from <a href="http://places.csail.mit.edu/">PLaces205 dataset</a> It has been used a pre-tained  AlexNet  model from <a href="https://github.com/BVLC/caffe/tree/master/models/bvlc_alexnet">Berkeley Vision and Learning Center</a>. Fine-tunning is applied after block 4.</p>

<img src="https://github.com/RicardDurall/CaffeInTorch/blob/master/Places/Imgaes/network.PNG" alt="CNN" style="width:304px;height:228px;">

<h4>Accuracy results of the CNN:</h4>

<img src="https://github.com/RicardDurall/CaffeInTorch/blob/master/Places/Imgaes/graph.PNG" alt="CNN" style="width:304px;height:228px;">

<h4>Euclidean distances:</h4>
<p> In this subsetion, we have computed the euclidean distance between the activations from 10000 trianing images and from 1000 testing images. We have repeated the experiment for several different activation layers</p>

<table style="width:100%">
  <tr>
    <td><img src="https://github.com/RicardDurall/CaffeInTorch/blob/master/Places/Imgaes/activation12.PNG" style="width:75px;height:76px;"></td>
    <td><img src="https://github.com/RicardDurall/CaffeInTorch/blob/master/Places/Imgaes/activation15.PNG" style="width:75px;height:76px;"></td>
  </tr>
    <tr>
    <td><img src="https://github.com/RicardDurall/CaffeInTorch/blob/master/Places/Imgaes/activation19.PNG" style="width:75px;height:76px;"></td>
    <td><img src="https://github.com/RicardDurall/CaffeInTorch/blob/master/Places/Imgaes/activation23.PNG" style="width:75px;height:76px;"></td>
  </tr>
</table>
