# Object Detection in an Urban Environment


## Project overview
Object detection is an essential component for autonomous vehicles because it allows the system to obtain a good understanding of the environment around it. Indeed, the vehicle may have to adapt its behavior to the objects it encounters: a pedestrian or a cyclist changes its trajectory much more easily and quickly than another vehicle.
Object detection is not an end in itself. It is one of the input elements of data fusion that builds a coherent and richest possible representation of the world.
The project aims to exploit the transfer capabilities of deep learning models. We will exploit pre-trained models on a general database of everyday objects (COCO) to specialize them on the recognition of vehicles, pedestrians and cyclists thanks to WAYMO's labeled data sets.


## Set up
To get around the limitation of GPU computing time and memory associated with the UDACITY workspace, I chose to work locally on my machine, equipped with a NVIDIA graphic card.
I first tried to use the docker container mentioned in the project instructions. 1st surprise: this container does not work! I didn't have any difficulty to install it but when launching the first code, some libraries can't be imported...
Quite unhappy to waste time on this type of problem that does not concern the heart of the project, I decided to install the components directly on my machine. The process was long and tedious. The compatibility problems of the different modules (CUDA, cuDNN, tensorflow, waymo.dataset...) are so numerous (I deplore that udacity doesn't provide any warning or any trustworthy source) that I wasted a lot of time on this subject. I am really dissatisfied with udacity's performance on this front!
I loaded the image segmentation already done in the udacity workspace. So I ignored the "prerequisite" part.

In order to use new models, especially the Faster RCNN, I had to install tensorrt. I met many compatibility problems between the versions of the different libraries (tensorrt, libnvinfer, libnvinfer-plugin...) to use for the project. I finally found a configuration working with tensorflow 2.8.0. I had to modify some scripts because the "legacy" and "experimental" modules were not present in my version (ema_optimizer.py, lars_optimizer.py, legacy_adam.py and optimizer_factory.py).
To use the other templates, I had to copy the pipeline configuration file (pipeline.config) provided in the zoo to the root of the project and run the script edit_config.py.
It is often necessary to correct the fine_tune_checkpoint_type parameter by assigning it the value 'detection'.



## Dataset
### Dataset analysis
I first tried to visualize the images used for learning, validation and testing.
It seemed to me that the data presented a great diversity of situations: roads of urban centers, highways, average roads. These different environments represent a diversity in terms of density of objects and location of objects in the scene (in the city, we find pedestrians in front of the vehicle - for example crossing the road - as well as on the side of the vehicle - walking on the sidewalk).
Different times of the day are present and therefore different sunlight conditions (from day to night). It is true that the night is less present than the day.
Finally, there are images of good weather, cloudy weather and rainy weather.
#### Train dataset
![plot](./images/discover_train_00.png)
![plot](./images/discover_train_01.png)

#### Validation dataset
![plot](./images/discover_val_00.png)
![plot](./images/discover_val_01.png)

#### Test dataset
![plot](./images/discover_test_00.png)
![plot](./images/discover_test_01.png)


I made a code that allows to extract the histograms of colorimetry of the images. For this to be useful, it would be necessary to be able to obtain the graphs for the different data sets but this takes a considerable amount of machine time and I was not able to do it. I did it only on 10 images randomly taken in the train, validation and test dataset.

The data are not easy to interpret due to the small number of cases used to produce the statistics.

#### Train dataset
![plot](./images/color_hist_train.png)
#### Validation dataset
![plot](./images/color_hist_val.png)
#### Test dataset
![plot](./images/color_hist_test.png)


### Cross validation
I have taken the split made on the udacity workspace. Nevertheless, here are the reasons why it seems justified to me.
The split was indeed performed on complete datasets corresponding to paths and not by making a random draw in the images from the set of paths. The different sets (as far as I could judge by randomly observing different sets of 10 images) are diverse (see the diversity points mentioned above).
I mentioned in the previous point the diversity of environments (urban, motorway, etc.). Nevertheless, after observing the first learning results and in particular the average quality of the predictions, especially for the "pedestrian" and "cyclist" classes, I have slightly changed my point of view. I therefore analysed the presence of the different classes in the different sets.
My analisis is limited to 500 000 batch per dataset for validation dataset and 100 000 for the training dataset. I have to do so because of the very long computation time needed.
#### Train dataset
![plot](./images/train_classes.png)
NOTA : my local configuration is broken. I can not produce this graph.
#### Validation dataset
![plot](./images/val_classes.png)



## Training
### Reference experiment
Here are the main points related to the reference experience:
    - model: SSD Resnet 50 640x640
    - optimizer : momentum
    - learning rate decay : cosine_decay
        - base: 0.4
    - number of steps: 2500
    - augnemtation strategies :
        - random_horizontal_flip
        - random_crop_image

The rather large value of the base learning rate leads to an important level of regularization_loss which moreover seems asymptotic at the end of the training. The regularization methods are used to allow the models to generalize better. With these parameters, our model seems to be in difficulty on the generalization.
The classification_loss and localization_loss do not really decrease during the learning process.
![plot](./images/loss_ref.png)
On the validation set, recall and precision are at very low levels.
#### Precision
![plot](./images/precision_ref.png)
#### Recall
![plot](./images/recall_ref.png)


### Improve on the reference
#### Improvement 1 (var_01)
As recommended in the instructions, I first played with the augmentation strategies.
Without touching the other parameters of the model (except the number of calculation steps), I used the following augmentation strategies:
    random_horizontal_flip
    random_adjust_brightness
    random_adjust_contrast
    random_adjust_hue
    random_adjust_saturation
    random_crop_image
I would have liked to be able to add "blur", "shear" and small angle rotations but unless I am mistaken, these strategies are not available in the API.
I thought it was relevant to keep the flip_horizontal because the objects we are looking for can come from all sides. If we had worked on signage (especially signs) this strategy might not have been relevant.
It seemed appropriate to use the different increases in luminosity, contrast and color to reinforce the variability of the weather conditions.
I kept the crop because it is an indirect way to play on the proximity of objects.

These modifications do not produce a very important effect. The extension of the calculation allows a small reduction of the localization_loss.
![plot](./images/loss_ref_01.png)

![plot](./images/precision_ref_01.png)

![plot](./images/recall_ref_00s.png)

##### Side note for augmentation
I was used the "augmentation notebook" to check the effect of different augmentation.
Here are some images with the augmentations that I have retain.
![plot](./images/augmented_00.png)

![plot](./images/augmented_01.png)

Here are some other images with the augmentations I don't have retain.
![plot](./images/augmented_02.png)
The rotation is too big in our context but you can not tune the rotation angle.

![plot](./images/augmented_03.png)
This patch is also unrealistic in our context. It looks more like a camera deffect. If you want to use patch, I think that it is better to use black patch.

![plot](./images/augmented_04.png)
I have retain augmentations playing with colors but the modification are here a bit unrealistic for a natural environment.

#### Improvement 2 (var_02)
I replaced the random_crop_image augmentation with random_crop_to_aspect_ratio to keep the proportions of the objects.
This slightly improves both the classification_loss and the localization_loss but we seem to lose a little bit of generalization capacity (regularization_loss).
Recall and precision are improving but remain at very low levels.
If we look at the "eval side by side" of tensorboard, we see false detections. Elements such as poles are confused by the model with cyclists. The performance on cars seems better. But there are less examples of cyclists in our train dataset than vehicules. It seems to be coherent.

![plot](./images/loss_01_02.png)

![plot](./images/precision_ref_02.png)

![plot](./images/recall_ref_00s.png)

#### Improvement 3 (var_03)
For this new test, I tried to decrease the learning_rate. The regularization_loss decreases but the changes are not significant on the classification_loss and the localization_loss.
The recall and the precision are also little impacted.

![plot](./images/loss_02_03.png)

![plot](./images/precision_ref_03.png)

![plot](./images/recall_ref_00s.png)

#### Improvement 4 (var_04)
So I tried to change the optimizer. I chose Adam instead of Momemtum. Again, the results are quite unaffected.
NOTA : In other series of tests I tried to play with the learning_rate evolution strategy (exponential, step). But there again I did not see any significant improvement.

![plot](./images/loss_02_03_04.png)

![plot](./images/precision_ref_03.png)

![plot](./images/recall_ref_00s.png)

#### Improvement 5 (var_30)
In this experiment, I extended the learning of the model to 40000 steps. The loss continues to decrease. It is this training that gives the best precision and recall but we remain at very low levels.

![plot](./images/loss_03_30.png)

![plot](./images/precision_ref_30.png)

![plot](./images/recall_ref_30.png)


#### Improvement 6 (var_40)
I tried to use another model than the Resnet 50 640x640 SSD. I used the Resnet 152 640x640 SSD model. The calculation times are significantly longer (x3 approximately) but the performance is not better.

![plot](./images/loss_30_40.png)
 
 I do not perform evaluation on this experiment.

#### Improvement 7 (var_50)
I switched to Taster RCNN coupled with Resnet 50 v1. Unfortunately I forgot to update my augmentation strategies. Only horizontal rotation was active.
The classification_loss and localization_loss do not appear on the same graphs anymore. If they are comparable with the previous experiments, the performance is much better. The total_loss is even divided by a factor of 3 compared to my best previous experiment (var_40).
The accuracy during validation improves. The same is true for recall.
Cyclists seem to be better captured by our model but there are still some false detections.

![plot](./images/loss_30_50.png)

![plot](./images/precision_ref_50s.png)

![plot](./images/recall_ref_50s.png)

#### Improvement 8 (var_51)
I decide to decrease the initial learning rate. Performance are a little better.

![plot](./images/loss_30_51.png)

![plot](./images/precision_ref_50s.png)

![plot](./images/recall_ref_50s.png)

#### Improvement 9 (var_52)
I update my augmentation strategies and I have extended the total steps number of the training.
This lead to the smallest loss I ever had.
Evaluation results are in line with the 2 previous training.
![plot](./images/loss_30_52.png)

![plot](./images/precision_ref_50s.png)

![plot](./images/recall_ref_50s.png)

I was unfortuntly unable to have animation script working. Here are some evaluation results with the reference model and the final model.

Here are the results of the reference model.
![plot](./images/result_ref_00.png)
![plot](./images/result_ref_01.png)
Here are the results of the final model on the same images.
![plot](./images/result_52_00.png)
![plot](./images/result_52_01.png)

Reference model do not capture anything in the 2 images. The final model also is missing some detection and produce some false detection (far objects).