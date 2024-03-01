# Import necessary modules
from argparse import ArgumentParser
import yaml
from pytorch_lightning import Trainer
from gan_module import AgingGAN  # Adjust the import based on your actual module structure
import torch

# Create an ArgumentParser object to handle command-line arguments
parser = ArgumentParser()
# Add a command-line argument '--config' with a default value and help message
parser.add_argument('--config', default='configs/aging_gan.yaml', help='Config to use for training')

# Define the main function
def main():
    # Parse the command-line arguments
    args = parser.parse_args()
    
    # Open the YAML configuration file specified in the '--config' argument
    with open(args.config) as file:
        # Load the YAML content into a dictionary using the PyYAML library
        config = yaml.load(file, Loader=yaml.FullLoader)
    
    # Print the loaded configuration for debugging purposes
    print(config)
    
    # Create an instance of the AgingGAN model using the loaded configuration
    model = AgingGAN(config)
    
    # Create a PyTorch Lightning Trainer instance with specified configurations
    trainer = Trainer(max_epochs=config['epochs'], auto_scale_batch_size='binsearch', gpus=1 if torch.cuda.is_available() else 0)
    
    # Train the model using the Trainer instance
    trainer.fit(model)

# Entry point of the script
if __name__ == '__main__':
    # Call the main function when the script is executed
    main()