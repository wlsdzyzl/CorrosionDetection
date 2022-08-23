from predictor import create_unet_predictor
from utils import load_config
def main():
    # Load and log experiment configuration
    config = load_config()
    print(config)
    # create trainer
    predictor = create_unet_predictor(config)
    predictor.predict()

main()