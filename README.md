## Running individual model files

To be able to run the files in **models** directory, change the directory to project root and run the following in the terminal:

`python -m models.<file_name>`

When you run the above command from your **automl_final** root directory, Python adds **automl_final** to *sys.path*. Because **automl_final** is now in *sys.path*, Python can easily resolve top-level imports like import utils.data_utils or import utils.model_utils.