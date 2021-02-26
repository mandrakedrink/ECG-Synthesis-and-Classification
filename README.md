# ECG-Synthesis-and-Classification
1D GAN for ECG Synthesis and 3 models: CNN, CNN with LSTM, and CNN with LSTM and Attention mechanism for ECG Classification

## Motivation 
ECG is widely used by cardiologists and medical practitioners for monitoring the cardiac health. The main problem
with manual analysis of ECG signals, similar to many other
time-series data, lies in difficulty of detecting and categorizing
different waveforms and morphologies in the signal. For a
human, this task is both extensively time-consuming and prone
to errors. Let's try to apply machine learning for this task.

## Data
Available [here](https://www.kaggle.com/shayanfazeli/heartbeat).

## Formulation of the problem:
Each signal should be labeled as one of the classes (**"Normal"**, **"Artial Premature"**, **"Premature ventricular contraction"**,**"Fusion of ventricular and normal"**, **"Fusion of paced and normal"**).

## Solution
Code with research and solution is available here - [1D GAN for ECG Synthesis](https://www.kaggle.com/polomarco/1d-gan-for-ecg-synthesis) and  here - [ECG Classification | CNN LSTM Attention mechanism](https://www.kaggle.com/polomarco/ecg-classification-cnn-lstm-attention-mechanism).

### Models

<p>
 <img src="https://64.media.tumblr.com/3499318b70fccebba076a3334bf7f0a4/4aeaf11bfa23b03f-59/s500x750/bae84e95d517fc08c23221afcf78acd194910e58.png" width="40%" height="100%">
 <img src="https://64.media.tumblr.com/e42e20eb2ec1aea3962c6ace63adf499/70877119c7741403-44/s1280x1920/b30b582fa0a23e4a9f7f5b35169d742bb615afc3.png" width="43%" height="100%">
</p>  
  
## GAN Results
<p>
 <img src="https://64.media.tumblr.com/dcfd4f40cdb257f033c4a8413d5b37df/8346578abfb69fc0-8d/s1280x1920/05d281b3fb71921c9cdb9500859088e3effdd103.png" width="80%" height="100%">

 <img src="https://64.media.tumblr.com/0e60c512867e5477a83a512387c6a892/be36c13a7534088a-32/s1280x1920/9ea416852c9b59465206083eca73538dd200b85a.png" width="80%" height="100%">
 <img src="https://64.media.tumblr.com/7ebadddc716c0028e26227dc4a57ffa2/485bd22c7e796deb-93/s1280x1920/543bb5fe34987eb1dd8d29f639c181af5247fe19.png" width="80%" height="100%">
</p>

## Classification Results
<p>
 <img src="https://64.media.tumblr.com/6d5197d1feedce9f1c2266cfbf4b9042/1901472907c7cef2-95/s1280x1920/f4d228eca6d3c3167199a4b06f51d1f074113d92.png" width="80%" height="100%">
 <img src="https://64.media.tumblr.com/dec2e645730bfe1e9c2ea33ee38afb97/1901472907c7cef2-33/s1280x1920/44edd463ddb319d35fc19e0dbdc6d7a4fb2f6253.png" width="80%" height="100%">
 <img src="https://64.media.tumblr.com/10e0744e620cdc7932a26e3b65a99431/1901472907c7cef2-a8/s1280x1920/079ad15698345eccb1e8ad76a4d0ad7d66ed20a7.png" width="80%" height="100%">
 <img src="https://64.media.tumblr.com/27fa12187b5dec2c863d6067dc4191d2/1901472907c7cef2-ba/s1280x1920/148766cb0c9619942316088a1704dafb42495fcf.png" width="80%" height="100%">
</p>
