# Import the necessary modules
from pmlb import fetch_data
from ydata_synthetic.synthesizers.regular import RegularSynthesizer
from ydata_synthetic.synthesizers import ModelParameters, TrainParameters

# Load data
data = fetch_data("adult")
num_cols = ["age", "fnlwgt", "capital-gain", "capital-loss", "hours-per-week"]
cat_cols = [
    "workclass",
    "education",
    "education-num",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
    "target",
]

# Define model and training parameters
ctgan_args = ModelParameters(batch_size=500, lr=2e-4, betas=(0.5, 0.9))
train_args = TrainParameters(epochs=10)

# Train the generator model
synth = RegularSynthesizer(modelname="ctgan", model_parameters=ctgan_args)
synth.fit(data=data, train_arguments=train_args, num_cols=num_cols, cat_cols=cat_cols)

# Generate 1000 new synthetic samples
synth_data = synth.sample(1000)
