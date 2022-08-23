from trainer import create_unet_trainer
from utils import load_config
def main():
    # Load and log experiment configuration
    config = load_config()
    print(config)
    # create trainer
    trainer = create_unet_trainer(config)
    trainer.train()

main()