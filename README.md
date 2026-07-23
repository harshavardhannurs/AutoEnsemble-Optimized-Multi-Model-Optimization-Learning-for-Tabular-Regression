## Running individual model files

To be able to run the files in **models** directory, change the directory to project root and run the following in the terminal:

`python -m models.<file_name>`

When you run the above command from your **automl_final** root directory, Python adds **automl_final** to *sys.path*. Because **automl_final** is now in *sys.path*, Python can easily resolve top-level imports like import utils.data_utils or import utils.model_utils.

## Using TabPFN

To use TabPFN,
  1. Open https://ux.priorlabs.ai in a browser and log in (or register)
  2. Accept the license on the Licenses tab
  3. Copy your API Key from https://ux.priorlabs.ai/account
  4. Create a **.env** file in the root directory of the project and add the following line:
    
        `API_TOKEN="<YOUR_API_KEY>"`