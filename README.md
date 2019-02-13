# Figure-Detection
Recognising if the image contains a figure or not

## Running the program
To check all supported options  
python3 figure_detection.py -h

1. To train a new model  
  python3 figure_detection.py --classes training.json 
  
2. To further train an existing model  
  python3 figure_detection.py --classes training.json --model existing_model.h5 --shape x,y
  
  Eg: python3 figure_detection.py --classes training.json --model existing_model.h5 --shape 1375,1022

3. To predict  
  python3 figure_detection.py --classes predict.json --model model.h5 --shape x,y --pred
