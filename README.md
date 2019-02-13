# Figure-Detection
Recognising if the image contains a figure or not

## Running the program
To check all supported options  
python3 figure_detection.py -h

1. To train a new model  
  python3 figure_detection.py --classes <training.json>  
  
2. To further train an existing model  
  python3 figure_detection.py --classes <training.json> --model <existing model> --shape <shape on which the model was trained>

3. To predict  
  python3 figure_detection.py --classes <predict.json> --model <model> --shape <shape> --pred
