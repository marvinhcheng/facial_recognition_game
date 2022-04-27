## Game Setup

Currently, the dlib library is only usable to me in a Conda environment so the facial recognition program is currently written in a jupyter notebook. Download all attached files and run the facial_rec.ipynb file in a Anaconda Prompt using the following command:

```
jupyter nbconvert --to notebook --inplace --execute facial_rec.ipynb
```
This wil create your "face_enc" file which contains the dictionary of names/faces. To add new faces to recognize, input training data into a new folder inside the "faces" folder. Make sure to name the new folder with the name of the new face.

Currently, the game's images are hand picked with the location of the face hardcoded.

To launch the game use the following command:
```
jupyter nbconvert --to notebook --inplace --execute game.ipynb
```

**Note:** The program can be launched from the notebook itself rather than command line if preferred.
