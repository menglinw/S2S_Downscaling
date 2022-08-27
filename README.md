# S2S_Downscaling
Develop a sequence to sequence downscaling method

## Project Structure
* data_loader
    * an object
        * load data
            * G5NR
            * MERRA2
            * Elevation
        * process data
            * 
        * output data for training
        * output data for testing

* model
    * directly define using TF
    * train with data from data_loader
    * save model
    
* evaluate
    * a batch of functions
        * use trained model to predict on test set
        * evaluate performance
