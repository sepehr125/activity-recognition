## Recognizing human activity from wearable accelerometers
[Dataset from UCI](http://archive.ics.uci.edu/ml/datasets/Activity+Recognition+from+Single+Chest-Mounted+Accelerometer)   

Signal from accelerometers is used to predict the wearer's activity among 7 classes:  

0. "Working at Computer"
0. "Standing Up, Walking and Going updown stairs"
0. "Standing"
0. "Walking"
0. "Going UpDown Stairs"
0. "Walking and Talking with Someone"
0. "Talking while Standing"        

Test error is around **7%** with 35 features, down from ~30% when trained on raw data.

You can tweak features in `featurize.py` and generate your own feature set.  
`python make_features.py my_features`  

Then train a model with:  
`python train_model.py my_features my_model`  

features.pkl is included to model with immediately.

Questions? Reach out to me at:
[sepehr125@gmail.com](mailto:sepehr125@gmail.com)
[@sepehr125](htpps://twitter.com/sepehr125)