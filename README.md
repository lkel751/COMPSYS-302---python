HELLO! Welcome to my DNN application, the gist of it is that starting it loads an app which allows you to download, train and test a deep neural network that will be able to recognize handwritten characters! 

An important note is for my dataset I didn't use the entire EMNIST dataset, this was due to it bieng too large and my computer bieng way too horrible and slow to effectively use it, so instead it just contains all digits and uppercase letters (0-9, A-Z) not (a-z). 
Also, python 3.8 was used as it was the only version that worked with all necessary tools to complete this project on my computer.

Once opening the main window you can click on 2 push buttons or use file to explore the various windows:

THE TRAINING WINDOW:
the most important window, here you can download the EMNIST dataset from online which contains hundreds of thousands of handwritten characters to use as a dataset to teach a model how to recognize handwriting, a download bar shows the progress of the download, aswell as a cancel button to cancel and close download or training. The training button will use this data to train up a model, use the slider to determine the EPOCH size of your model, the higher the epoch the longer the download but the more accurate the model.

THE VIEW WINDOW:
Here you can enter a valid character (0-9, A-Z) and the model will spit out 100 images of said character.

TESTING WINDOW:
the most fun window! Here you draw the character with the canvas on the right and desired brush size, click predict and the model will attempt to predict what character it is. The probability and predicted character are shown in the box at the bottom right of the screen.

this youtuber was essential in creating this application as I spent tens of hours loading and crashing my computer with different variations of creating models, as even using skilearn to shorten the data necessary, a lot of data must already be loaded in order to use it. Altthough it is less then the full database, it was my only option and I am still happy with it
https://www.youtube.com/watch?v=kOF2Lp_GbkQ

Also used this guy for the canvas widget in the testing class.
https://www.youtube.com/watch?v=qEgyGyVA1ZQ

all most other features contained in the code I used this playlist
https://www.youtube.com/playlist?list=PLzMcBGfZo4-lB8MZfHPLTEHO9zJDDLpYj

all other stuff is in the bibliography of the report, specifics about the application are also in here, with the finer details commented in the actual code.

