# Star_Classifier_Analysis
Space object classifier with a data analysis and a visualisation.
## About the dataset:
File 'star_classification.csv'contains of 100 000 observations of a space
and an every observation is described by 18 feature columns which a 14th is a
class column that defines if a observation is either a star, galaxy or quasar.

Column - informations:

1.'obj_ID' - Object Identifier, the unique value that identifies the object in the image catalog used by the CAS

2.'alpha' - Right Ascension angle (at J2000 epoch)

3.'delta' - Declination angle (at J2000 epoch)

4.'u' - Ultraviolet filter in the photometric system


5.'g' - Green filter in the photometric system

6.'r' - Red filter in the photometric system

7.'i' - Near Infrared filter in the photometric system

8.'z' - Infrared filter in the photometric system

9.'run_ID' - Run Number used to identify the specific scan

10.'rereun_ID' - Rerun Number to specify how the image was processed

11.'cam_col' - Camera column to identify the scanline within the run

12.'field_ID' - Field number to identify each field

13.'spec_obj_ID' - Unique ID used for optical spectroscopic objects (this means that 2 different observations with the same spec_obj_ID must share the output class)

14.'class' - object class (galaxy, star or quasar object)

15.'redshift' - redshift value based on the increase in wavelength

16.'plate' - plate ID, identifies each plate in SDSS

17.'MJD' - Modified Julian Date, used to indicate when a given piece of SDSS data was taken

18.'fiber_ID' - fiber ID that identifies the fiber that pointed the light at the focal plane in each observation

## Short analysis of the data and exploratory data analysis:
File 'RawData.py' contains a short analysis to get a brief info 
about the data like a distribution, amount, statistic informations...
- Possible patterns of 'class' objects using visualisations:
  ![alpha_delta_basic](https://github.com/claudia13062013/Star_Classifier_Analysis/assets/97663507/c3bea1c6-0bdc-4af1-ada1-f080f438b71f)
- using library 'astropy' to more clear astronomy visualisation:
  ![apha_delta_sky_visual](https://github.com/claudia13062013/Star_Classifier_Analysis/assets/97663507/dbd2078e-2f4d-4505-a932-3e03558ee7d4)

- amount of an every class :
  ![class_amount](https://github.com/claudia13062013/Star_Classifier_Analysis/assets/97663507/a9c519a0-7e23-4c50-9684-309c19e3c343)

File 'AnalysisEDAData.py' contains a deeper analysis with the exploration of correlations
and patterns
- visualisation of an every class on the sky with a feature 'alpha' and 'delta':
   ![plot_every_object](https://github.com/claudia13062013/Star_Classifier_Analysis/assets/97663507/32802f10-0143-4763-8184-7a751a6928a8)
-correlations: Pearson's correlation:
-quasar:
![heatmap_corr_qso](https://github.com/claudia13062013/Star_Classifier_Analysis/assets/97663507/259ace78-9c87-4ded-845e-47a34032d37f)

-star:
![heatmap_corr_stars](https://github.com/claudia13062013/Star_Classifier_Analysis/assets/97663507/64483acf-cb7b-46f9-a726-98f497241a2c)

-galaxy:
![heatmap_corr_galaxy](https://github.com/claudia13062013/Star_Classifier_Analysis/assets/97663507/1924532f-27ff-4f9c-8e0b-48f5eda40988)

Spearman's correlation for 'star':
![heatmap_corr_stars_spearman](https://github.com/claudia13062013/Star_Classifier_Analysis/assets/97663507/abd76f16-7e1e-450a-b08b-c0a7e1700c62)

- Distribution of redshifts with quasars and redshifts with stars and galaxies:
  ![class_redshift_plot](https://github.com/claudia13062013/Star_Classifier_Analysis/assets/97663507/6ed1929c-d026-47af-93df-763d992a1e0b)

## Bulding ML model :
File 'ML_model.py' contains a machine learning model that classifie an observation with 98.5% accuracy

- taking all informations from the analysis some columns could be dropped to make a model
  #df.drop(['obj_ID', 'delta', 'alpha',  'run_ID', 'rerun_ID', 'cam_col', 'field_ID', 'spec_obj_ID', 'fiber_ID']...
  
- oversampling was used to make an equal amount of every class
  ![classes_oversampling](https://github.com/claudia13062013/Star_Classifier_Analysis/assets/97663507/06ebd645-8fac-404d-ac07-27153fa181d3)


- the Random Forest Classifier was used to train a model and make an accurate classifier

- test results was made with a conffusion matrix and a cross validation

All test results are located in file 'run.txt'.


 






