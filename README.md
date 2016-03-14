## Activity Recognition from Chest-Mounted Accelerometer
### (Dataset from UCI)[http://archive.ics.uci.edu/ml/datasets/Activity+Recognition+from+Single+Chest-Mounted+Accelerometer].  

Signal from accelerometers is used to predict the wearer's activity among 7 classes:  
0. "Working at Computer"
0. "Standing Up, Walking and Going updown stairs"
0. "Standing"
0. "Walking"
0. "Going UpDown Stairs"
0. "Walking and Talking with Someone"
0. "Talking while Standing"        

Test set accuracy is around 93% with 35 features, up from ~70% when trained on raw data.

You can train your own random forest model. Simply run:  
`python train_rf.py features.pkl model_name.pkl`  

You can tweak features in `featurize.py` and generate your own feature set.  
`python featurize.py 5 0.5 outputfile.pkl`  
This will window over the raw data in 5 second intervals with 0.5 overlap of windows (i.e. 2.5s),
generate features (e.g. Fourier transform peaks, etc.), and save them as `outputfile.pkl`.